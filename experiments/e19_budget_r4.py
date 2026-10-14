#!/usr/bin/env python
"""E19: where is the error NOW, after round 2's rules and round 3's parameters?

`budget.py` decomposed the v2-equivalent model into normal / missing-aircraft-type /
invisible. Both anomalous populations have since been attacked (delta regime model,
extreme-proxy override), so the question that matters in round 4 is different: **inside the
"normal" 53.5%, which sub-population carries the squared error?**

Same discipline as budget.py -- weighted through `src.validation`'s composition so every
number is comparable to the submission log in SKILLS.md.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.features import model_features
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)
from src.validation import mirrored_split, ranking_composition, report, weighted_rmse

PARAMS_PATH = Path("data/processed/lgbm_best_params_r3.json")
INT_P = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}

raw = json.loads(PARAMS_PATH.read_text())
p = {k: (int(v) if k in INT_P else v) for k, v in raw.items() if not k.startswith("_")}
n_est = p.pop("n_estimators")
params = dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1,
              subsample_freq=1, random_state=0, n_estimators=n_est, **p)

comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
d = d.merge(pd.read_parquet("data/processed/train_surface.parquet"),
            on="MVT_ID_mvt", how="left", validate="one_to_one")
nums, cats = model_features(d)
for c in cats:
    d[c] = d[c].astype("string").fillna("__NA__").astype("category")
feats = nums + cats

is_train, is_val, cell = mirrored_split(d)
tr, val = d[is_train & trainable_mask(d)], d[is_val].copy()
y, yv = tr["TAXITIME_SEC_mvt"], val["TAXITIME_SEC_mvt"].to_numpy()
print(f"fit {len(tr):,} | val {len(val):,} | {len(feats)} features | {n_est} iterations", flush=True)

t0 = time.time()
model = lgb.LGBMRegressor(**params).fit(tr[feats], y)
print(f"trained in {time.time()-t0:.0f}s", flush=True)
pred = model.predict(val[feats])
pred, n1 = apply_regime_model(val, pred, fit_regime_model(d[is_train], feats))
pred, n2 = apply_extreme_proxy_override(val, pred)
print(f"regime rows {n1}, override rows {n2}")

vcell, vapt = cell[is_val], val["ADEP_mvt"]
trim = trainable_mask(val).to_numpy()
r = report(yv, pred, val, comp, vcell, trimmed_mask=trainable_mask(val))
print(f"\n*** weighted RMSE  full {r['rmse_full']:.2f}s   trimmed {r['rmse_trimmed']:.2f}s ***\n")
np.save("data/processed/exp_cache/r4_pred.npy", pred)
val[["MVT_ID_mvt"]].to_parquet("data/processed/exp_cache/r4_val_ids.parquet")

cellw = comp.set_index(["slice", "airport"])["weight"]
sq = (pred - yv) ** 2
e = pd.DataFrame({"slice": vcell.to_numpy(), "airport": vapt.to_numpy().astype(str), "sq": sq,
                  "y": yv, "pred": pred}, index=val.index)
# contribution to the weighted mean square: cell weight x cell mean square, renormalised
cellms = e.groupby(["slice", "airport"], observed=True)["sq"].mean()
celln = e.groupby(["slice", "airport"], observed=True)["sq"].size()
w = cellw.reindex(cellms.index).fillna(0.0)
denom = float((w * cellms).sum())
print("=== per-cell, and its share of the weighted MSE ===")
tbl = pd.DataFrame({"n": celln, "rmse": np.sqrt(cellms), "weight": w,
                    "share": (w * cellms) / denom}).sort_values("share", ascending=False)
print(tbl.assign(rmse=tbl.rmse.round(1), weight=tbl.weight.round(4),
                 share=(tbl.share * 100).round(1)).to_string())

def share_of(mask, label):
    """Share of the weighted MSE carried by a subset of rows, and its own RMSE."""
    m = np.asarray(mask)
    if m.sum() == 0:
        return
    contrib = e[m].groupby(["slice", "airport"], observed=True)["sq"].sum() / celln
    ww = cellw.reindex(contrib.index).fillna(0.0)
    s = float((ww * contrib).sum()) / denom
    print(f"  {label:<44} n={int(m.sum()):>7,} ({m.mean():6.2%})  "
          f"rmse {np.sqrt(sq[m].mean()):7.1f}s  share {s:6.1%}")

noact = val["dq_no_aircraft_type"].to_numpy()
impl = val["dq_implausible_target"].to_numpy()
normal = ~(noact | impl)
print("\n=== round-2 populations, re-measured under the current model ===")
share_of(normal, "normal")
share_of(noact, "missing AIRCRAFT_TYPE_mvt")
share_of(impl & ~noact, "invisible broken label")

print("\n=== inside the normal rows: where does ITS error sit? ===")
nsq = np.where(normal, sq, 0.0)
order = np.argsort(-nsq)
for frac in (0.001, 0.01, 0.05, 0.10):
    k = int(frac * normal.sum())
    m = np.zeros(len(val), bool); m[order[:k]] = True
    share_of(m, f"worst {frac:.1%} of normal rows by squared error")

print("\n=== candidate splits of the normal rows ===")
has3 = val["AOBT_3_flt"].notna().to_numpy()
share_of(normal & ~has3, "normal, AOBT_3_flt MISSING (no ADS-B proxy)")
share_of(normal & has3, "normal, AOBT_3_flt present")
if "offblock_agree_n" in val.columns:
    ag = val["offblock_agree_n"].to_numpy()
    for k in range(5):
        share_of(normal & (ag == k), f"normal, offblock_agree_n == {k}")
for lo, hi in [(0, 600), (600, 1200), (1200, 1800), (1800, 2700), (2700, 3600), (3600, 1e9)]:
    share_of(normal & (yv >= lo) & (yv < hi), f"normal, true taxi in [{lo}, {hi}) s")

print("\n=== per-cell bias (mean signed residual, pred - y), normal rows only ===")
b = e[normal].groupby(["slice", "airport"], observed=True).apply(
    lambda g: pd.Series({"n": len(g), "bias": (g.pred - g.y).mean(),
                         "med_bias": (g.pred - g.y).median()}), include_groups=False)
print(b.round(1).to_string())
