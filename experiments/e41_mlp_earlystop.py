#!/usr/bin/env python
"""E41: the embedding MLP, trained properly — early stopping, refit, seed bagging.

`e39` gave the MLP a fixed 15-epoch OneCycle budget chosen for speed, and it overfitted
visibly: final train MSE 0.1809 (~184 s training RMSE) against 242.74 s validation trimmed. It
still scored 284.20 full and blended with LightGBM for −1.22 s at w=0.75, with a residual
correlation of **0.9003** — genuinely decorrelated, which is where the blend value comes from.

This trains it the way the rest of the repo trains things:

1. **Early stopping on an inner split** — the last training month, held out of the fit, exactly
   as `scripts/tune_lgbm.py` does. The scored validation slice is never consulted for the
   stopping decision; doing so is the defect round 3 had to fix in the LightGBM sweep.
2. **Refit on the whole training window** at the chosen epoch count, so the scored model has
   seen all six months rather than five.
3. **Seed bagging** over three runs, as the shipped LightGBM does (worth −0.84 s there, and
   networks are higher-variance than boosted trees, so it should matter at least as much).

`ReduceLROnPlateau` rather than OneCycle: a schedule tied to a fixed epoch count cannot be
combined with early stopping without changing its own shape, which would confound the two.
"""
from __future__ import annotations
import argparse, copy, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.cleaning import trainable_mask
from src.config_features import CONFIG_CATEGORICALS
from src.features import model_features
from src.offblock_gaps import add_offblock_gaps
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)
from src.validation import TRAIN_MONTHS, mirrored_split, ranking_composition, report

OFFSET = 5.73
CONFIG_PREFIXES = ("depcfg_", "arrcfg_", "stand_turnaround", "stand_prev_arr")
CACHE = Path("data/processed/exp_cache")

ap = argparse.ArgumentParser()
ap.add_argument("--max-epochs", type=int, default=40)
ap.add_argument("--patience", type=int, default=4)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(args.threads)

print("loading...", flush=True)
comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
for extra in ("train_surface", "train_config", "train_osm"):
    d = d.merge(pd.read_parquet(f"data/processed/{extra}.parquet"),
                on="MVT_ID_mvt", how="left", validate="one_to_one")
d, _ = add_offblock_gaps(d)
nums, cats = model_features(d)
cats = list(cats) + [c for c in CONFIG_CATEGORICALS if c in d.columns]
nums = [c for c in nums if c not in cats]

is_train, is_val, cell = mirrored_split(d)
trainable = trainable_mask(d)
tr_mask = (is_train & trainable).to_numpy()
va_mask = is_val.to_numpy()
# the stopping month comes out of the TRAINING window; the scored slice is never consulted
inner = ((d["MVT_TIME_UTC_mvt"].dt.month == max(TRAIN_MONTHS)).to_numpy()) & tr_mask
outer = tr_mask & ~inner

codes, cards = [], []
for c in cats:
    col = d[c].astype("string").fillna("__NA__")
    cat = pd.Categorical(col)
    codes.append(cat.codes.astype("int64"))
    cards.append(len(cat.categories))
    d[c] = cat
X_cat = np.stack(codes, axis=1)

Xn = d[nums].to_numpy(dtype="float32")
miss = np.isnan(Xn)
med = np.nanmedian(Xn[tr_mask], axis=0)
med = np.where(np.isfinite(med), med, 0.0).astype("float32")
Xn = np.where(miss, med, Xn)
mu, sd = Xn[tr_mask].mean(axis=0), Xn[tr_mask].std(axis=0)
sd = np.where(sd < 1e-6, 1.0, sd).astype("float32")
Xn = np.clip(((Xn - mu) / sd).astype("float32"), -10, 10)
X_num = np.concatenate([Xn, miss.astype("float32")], axis=1)

y = d["TAXITIME_SEC_mvt"].to_numpy(dtype="float32")
y_mu, y_sd = float(y[tr_mask].mean()), float(y[tr_mask].std())
ys = ((y - y_mu) / y_sd).astype("float32")
print(f"{len(cats)} cats (max card {max(cards):,}) | {X_num.shape[1]} numeric | "
      f"outer {outer.sum():,} | inner-stop {inner.sum():,} | val {va_mask.sum():,}\n", flush=True)


class EmbMLP(nn.Module):
    def __init__(self, cards, n_num, hidden=(512, 256, 128), p=0.15):
        super().__init__()
        dims = [min(64, (c + 1) // 2) for c in cards]
        self.embs = nn.ModuleList([nn.Embedding(c, dd) for c, dd in zip(cards, dims)])
        layers, prev = [], sum(dims) + n_num
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, xc, xn):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
        return self.mlp(torch.cat(e + [xn], dim=1)).squeeze(1)


def loader(mask, shuffle, bs):
    return DataLoader(TensorDataset(torch.from_numpy(X_cat[mask]),
                                    torch.from_numpy(X_num[mask]),
                                    torch.from_numpy(ys[mask])),
                      batch_size=bs, shuffle=shuffle, drop_last=shuffle)


@torch.no_grad()
def predict(model, mask, bs=8192):
    model.eval()
    out = []
    idx = np.flatnonzero(mask)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        out.append(model(torch.from_numpy(X_cat[j]), torch.from_numpy(X_num[j])).numpy())
    return np.concatenate(out)


def train(mask, seed, epochs=None, stop_mask=None):
    """Train on `mask`. With `stop_mask`, early-stop on it and return the best epoch."""
    torch.manual_seed(seed)
    model = EmbMLP(cards, X_num.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    lossf = nn.MSELoss()
    dl = loader(mask, True, args.batch)
    best, best_ep, bad, best_state = np.inf, 0, 0, None
    for ep in range(1, (epochs or args.max_epochs) + 1):
        model.train(); t0 = time.time(); tot = n = 0
        for xc, xn, yy in dl:
            opt.zero_grad()
            loss = lossf(model(xc, xn), yy)
            loss.backward(); opt.step()
            tot += float(loss.detach()) * len(yy); n += len(yy)
        msg = f"  seed {seed} epoch {ep:>2}  train {tot/n:.4f}"
        if stop_mask is not None:
            v = float(np.mean((predict(model, stop_mask) - ys[stop_mask]) ** 2))
            sched.step(v)
            msg += f"  inner {v:.4f}"
            if v < best - 1e-4:
                best, best_ep, bad = v, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
        print(f"{msg}  ({time.time()-t0:.0f}s)", flush=True)
        if stop_mask is not None and bad >= args.patience:
            print(f"  early stop at epoch {ep}; best was {best_ep} (inner {best:.4f})", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_ep or (epochs or args.max_epochs))


print("=== phase 1: early stopping on the inner month ===", flush=True)
_, best_ep = train(outer, seed=0, stop_mask=inner)
print(f"\n=== phase 2: refit on the full training window for {best_ep} epochs, "
      f"{args.seeds} seeds ===", flush=True)
preds = []
for s in range(args.seeds):
    m, _ = train(tr_mask, seed=s, epochs=best_ep)
    preds.append(predict(m, va_mask) * y_sd + y_mu)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "nn2_val_pred.npy", np.mean(preds, axis=0))
    print(f"  seed {s} done, {len(preds)} in the bag", flush=True)
pred_nn = np.mean(preds, axis=0)

val = d[is_val].copy()
yv = val["TAXITIME_SEC_mvt"].to_numpy()
vcell, trimmed = cell[is_val], trainable_mask(val)
regime = fit_regime_model(d[is_train], [c for c in nums + cats if not c.startswith(CONFIG_PREFIXES)])

def score(p, label):
    p = np.asarray(p, dtype="float64").copy()
    p, _ = apply_regime_model(val, p, regime)
    p, _ = apply_extreme_proxy_override(val, p)
    r = report(yv, p, val, comp, vcell, trimmed_mask=trimmed)
    print(f"{label:<34} full {r['rmse_full']:>7.2f}  trimmed {r['rmse_trimmed']:>7.2f}  "
          f"board {r['rmse_full']+OFFSET:>7.2f}")
    return r

print()
score(pred_nn, f"MLP, early-stopped, {args.seeds}-seed bag")
print("\n(e39's fixed-budget MLP: 284.20 / 242.74;  LightGBM v5 on this box: 273.72 / 230.21)")

lgb_cache = CACHE / "lgb_v5_val_pred.npy"
if lgb_cache.exists():
    pred_lgb = np.load(lgb_cache)
    norm = trimmed.to_numpy()
    rho = np.corrcoef(pred_lgb[norm] - yv[norm], pred_nn[norm] - yv[norm])[0, 1]
    print(f"\nresidual correlation with LightGBM: {rho:.4f}  (e39's MLP: 0.9003)")
    print(f"\n{'weight on LightGBM':<34}")
    for w in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0):
        score(w * pred_lgb + (1 - w) * pred_nn, f"blend w = {w:.2f}")
