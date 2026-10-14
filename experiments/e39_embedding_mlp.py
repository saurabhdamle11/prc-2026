#!/usr/bin/env python
"""E39: an embedding MLP — the one model class never tried on this problem.

The case for it is narrow but real. Trees split one axis at a time and handle high-cardinality
categoricals by partitioning levels; a network gives each level a learned vector. We have
`stand_full` (2,723 levels), `ADES_mvt` (1,569), `AIRCRAFT_OPERATOR_flt` (677) and
`AIRCRAFT_TYPE_mvt` (270), and `cat_smooth` was the **only** hyperparameter that moved in
round 4 (`SKILLS.md` §11) — which says categorical handling is a live lever here.

The case against is the stronger one. On tabular data of this size gradient-boosted trees
usually win, and more importantly our trimmed RMSE (231.5) sits far above the measured oracle
(152.3) because of **unobserved** information — the apron hold, proven unobservable in
`e34`. No architecture manufactures information it was not given.

So the realistic value is **ensemble diversity**, not standalone accuracy. CatBoost blended for
only 0.9 s because it makes errors correlated with LightGBM's; an MLP should decorrelate more.

Protocol is identical to every other experiment: the mirrored split, the same post-processing,
weighted RMSE. Stop conditions are stated up front — if it is more than 50 s worse alone, the
blend will not save it.
"""
from __future__ import annotations
import argparse, json, sys, time
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
from src.validation import mirrored_split, ranking_composition, report

OFFSET = 5.73
CONFIG_PREFIXES = ("depcfg_", "arrcfg_", "stand_turnaround", "stand_prev_arr")

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(0)

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

# --- categoricals -> integer codes, one shared vocabulary so train and val agree -----------
codes, cards = [], []
for c in cats:
    col = d[c].astype("string").fillna("__NA__")
    cat = pd.Categorical(col)
    codes.append(cat.codes.astype("int64"))
    cards.append(len(cat.categories))
    d[c] = cat            # the regime branch fits LightGBM on these and needs a real dtype
X_cat = np.stack(codes, axis=1)

# --- numerics: median-impute with a missingness indicator, then standardise ----------------
Xn = d[nums].to_numpy(dtype="float32")
miss = np.isnan(Xn)
med = np.nanmedian(Xn[tr_mask], axis=0)
med = np.where(np.isfinite(med), med, 0.0).astype("float32")
Xn = np.where(miss, med, Xn)
mu = Xn[tr_mask].mean(axis=0)
sd = Xn[tr_mask].std(axis=0)
sd = np.where(sd < 1e-6, 1.0, sd).astype("float32")
Xn = ((Xn - mu) / sd).astype("float32")
Xn = np.clip(Xn, -10, 10)                     # a few proxies have 24h outliers
X_num = np.concatenate([Xn, miss.astype("float32")], axis=1)

y = d["TAXITIME_SEC_mvt"].to_numpy(dtype="float32")
y_mu, y_sd = float(y[tr_mask].mean()), float(y[tr_mask].std())
print(f"{len(cats)} categoricals (max card {max(cards):,}) | {X_num.shape[1]} numeric inputs "
      f"| fit {tr_mask.sum():,} | val {va_mask.sum():,}", flush=True)


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


dev = "cpu"
model = EmbMLP(cards, X_num.shape[1]).to(dev)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

tr_ds = TensorDataset(torch.from_numpy(X_cat[tr_mask]), torch.from_numpy(X_num[tr_mask]),
                      torch.from_numpy(((y[tr_mask] - y_mu) / y_sd)))
dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl))
lossf = nn.MSELoss()

for ep in range(args.epochs):
    model.train(); t0 = time.time(); tot = n = 0
    for xc, xn, yy in dl:
        opt.zero_grad()
        loss = lossf(model(xc, xn), yy)
        loss.backward(); opt.step(); sched.step()
        tot += float(loss) * len(yy); n += len(yy)
    print(f"  epoch {ep+1:>2}/{args.epochs}  train mse {tot/n:.4f}  ({time.time()-t0:.0f}s)", flush=True)

model.eval()
preds = []
with torch.no_grad():
    for i in range(0, int(va_mask.sum()), 8192):
        xc = torch.from_numpy(X_cat[va_mask][i:i+8192])
        xn = torch.from_numpy(X_num[va_mask][i:i+8192])
        preds.append(model(xc, xn).numpy())
pred_nn = np.concatenate(preds) * y_sd + y_mu
Path("data/processed/exp_cache").mkdir(parents=True, exist_ok=True)
np.save("data/processed/exp_cache/nn_val_pred.npy", pred_nn)   # save BEFORE scoring
torch.save({"state": model.state_dict(), "cards": cards, "n_num": X_num.shape[1],
            "y_mu": y_mu, "y_sd": y_sd}, "data/processed/nn_model.pt")
print(f"raw MLP prediction cached ({len(pred_nn):,} rows)", flush=True)

val = d[is_val].copy()
yv = val["TAXITIME_SEC_mvt"].to_numpy()
vcell, trimmed = cell[is_val], trainable_mask(val)
feats_for_regime = [c for c in nums + cats if not c.startswith(CONFIG_PREFIXES)]
regime = fit_regime_model(d[is_train], feats_for_regime)

def score(p, label):
    p = np.asarray(p, dtype="float64")
    p, _ = apply_regime_model(val, p, regime)
    p, _ = apply_extreme_proxy_override(val, p)
    r = report(yv, p, val, comp, vcell, trimmed_mask=trimmed)
    print(f"{label:<34} full {r['rmse_full']:>7.2f}  trimmed {r['rmse_trimmed']:>7.2f}  "
          f"board {r['rmse_full']+OFFSET:>7.2f}")
    return r

print()
score(pred_nn, "embedding MLP alone")
print("\n(reference: LightGBM v5 3-seed = 275.27 full / 230.22 trimmed)")
print("blend weights are measured in e40 once a LightGBM prediction is cached alongside this.")
