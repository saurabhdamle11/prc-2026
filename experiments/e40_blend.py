#!/usr/bin/env python
"""E40: does blending the embedding MLP with LightGBM help?

`e39` scored the MLP at 284.20 full / 242.74 trimmed against LightGBM v5's 275.27 / 230.22 —
**8.9 s worse standalone**, which is close enough for a blend to be worth measuring. Round 2's
CatBoost was a comparable 8.4 s behind and blended for only **0.9 s**, but CatBoost is another
tree ensemble and its errors are highly correlated with LightGBM's. An MLP with learned
embeddings is a genuinely different function class, so the interesting quantity is the
**correlation of the two residuals**, not the standalone gap.

Caches the LightGBM validation predictions so any future blend costs nothing to re-measure.

.. warning::
   The blend weight is chosen on the same slice it is scored on. Round 2 recorded exactly this
   caveat for the CatBoost blend, and the honest expectation is below whatever the grid's best
   point shows. A gain that only appears at one finely-tuned weight is not a gain.
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.config_features import CONFIG_CATEGORICALS
from src.features import model_features
from src.offblock_gaps import add_offblock_gaps
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)
from src.validation import mirrored_split, ranking_composition, report

OFFSET = 5.73
CONFIG_PREFIXES = ("depcfg_", "arrcfg_", "stand_turnaround", "stand_prev_arr")
CACHE = Path("data/processed/exp_cache")
LGB_CACHE = CACHE / "lgb_v5_val_pred.npy"

raw = json.loads(Path("data/processed/lgbm_best_params_r3.json").read_text())
INT_P = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}
p = {k: (int(v) if k in INT_P else v) for k, v in raw.items() if not k.startswith("_")}
n_est = p.pop("n_estimators")
PARAMS = dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1, subsample_freq=1,
              n_estimators=n_est, cat_smooth=100.0, **p)

comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
for extra in ("train_surface", "train_config", "train_osm"):
    d = d.merge(pd.read_parquet(f"data/processed/{extra}.parquet"),
                on="MVT_ID_mvt", how="left", validate="one_to_one")
d, _ = add_offblock_gaps(d)
nums, cats = model_features(d)
cats = list(cats) + [c for c in CONFIG_CATEGORICALS if c in d.columns]
for c in cats:
    d[c] = d[c].astype("string").fillna("__NA__").astype("category")
feats = nums + cats

is_train, is_val, cell = mirrored_split(d)
trainable = trainable_mask(d)
tr = d[is_train & trainable]
val = d[is_val].copy()
regime = fit_regime_model(d[is_train], [c for c in feats if not c.startswith(CONFIG_PREFIXES)])
vcell, trimmed = cell[is_val], trainable_mask(val)
y, yv = tr["TAXITIME_SEC_mvt"], val["TAXITIME_SEC_mvt"].to_numpy()

if LGB_CACHE.exists():
    pred_lgb = np.load(LGB_CACHE)
    print(f"LightGBM validation predictions loaded from cache ({len(pred_lgb):,} rows)")
else:
    print(f"fitting LightGBM v5, 3 seeds, {len(feats)} features...", flush=True)
    preds = []
    for s in (0, 1, 2):
        t0 = time.time()
        m = lgb.LGBMRegressor(**PARAMS, random_state=s).fit(tr[feats], y)
        preds.append(m.predict(val[feats])); del m; gc.collect()
        print(f"  seed {s} ({time.time()-t0:.0f}s)", flush=True)
    pred_lgb = np.mean(preds, axis=0)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(LGB_CACHE, pred_lgb)

pred_nn = np.load(CACHE / "nn_val_pred.npy")
assert len(pred_nn) == len(pred_lgb) == len(val)


def score(p, label):
    p = np.asarray(p, dtype="float64").copy()
    p, _ = apply_regime_model(val, p, regime)
    p, _ = apply_extreme_proxy_override(val, p)
    r = report(yv, p, val, comp, vcell, trimmed_mask=trimmed)
    print(f"{label:<30} full {r['rmse_full']:>7.2f}  trimmed {r['rmse_trimmed']:>7.2f}  "
          f"board {r['rmse_full']+OFFSET:>7.2f}")
    return r


# how independent are the two models' mistakes? that is what decides a blend
norm = trainable_mask(val).to_numpy()
e_lgb, e_nn = pred_lgb[norm] - yv[norm], pred_nn[norm] - yv[norm]
print(f"\nresidual correlation on normal rows: {np.corrcoef(e_lgb, e_nn)[0,1]:.4f}")
print("(round 2's CatBoost blend was worth 0.9 s; the lower this is, the more a blend can help)\n")

print(f"{'weight on LightGBM':<30} {'full':>12} {'trimmed':>16}")
best = None
for w in (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0):
    r = score(w * pred_lgb + (1 - w) * pred_nn, f"w = {w:.2f}")
    if best is None or r["rmse_full"] < best[1]:
        best = (w, r["rmse_full"], r["rmse_trimmed"])
print(f"\nbest w={best[0]:.2f}: {best[1]:.2f} full / {best[2]:.2f} trimmed")
print(f"vs LightGBM alone: {best[1] - 275.27:+.2f}s full / {best[2] - 230.22:+.2f}s trimmed")
print("\nShip only if the gain clearly exceeds ~1 s AND holds on the trimmed number — the "
      "weight is chosen on the slice it is scored on.")
