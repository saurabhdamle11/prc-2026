#!/usr/bin/env python
"""E32: what is a self-computed off-block ACTUALLY worth? Simulate it before buying it.

Before spending money and a week of the remaining window on ADS-B extraction, size the prize
using data already in hand. Training rows carry `BLOCK_TIME_UTC_mvt`, so a detector of any
given quality can be *simulated*: take the truth, corrupt it with noise of scale `sigma`, hide
it on a fraction of rows to mimic imperfect ADS-B ground coverage, and hand the result to the
real model as one extra feature.

    proxy_taxi_adsb = MVT_TIME - (BLOCK_TIME + N(0, sigma))     on `coverage` of rows, else NaN

The resulting validated RMSE is what a real detector of that quality would buy. This is a
**sizing simulation, not a model** -- it uses the target's other half and could never ship. Its
only purpose is to decide whether the ADS-B project is worth starting.

Reference points it has to be read against:
  * `AOBT_3_flt`, the estimator the model leans on today, has a weighted RMSE of **422.9 s**
    against the true BLOCK_TIME (`e31`).
  * v4 ships at 276.50 offline / 282.23 board; the protocol offset is **+5.73 s** (`SKILLS.md` §7).
  * The leader is at **224**, and v4's trimmed RMSE -- every anomalous row solved perfectly --
    is 237.2 on the board, so the tail cannot close that gap (`SKILLS.md` §11).
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.config_features import CONFIG_CATEGORICALS
from src.features import model_features
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)
from src.validation import mirrored_split, ranking_composition, report

OFFSET = 5.73
CONFIG_PREFIXES = ("depcfg_", "arrcfg_", "stand_turnaround", "stand_prev_arr")
SIM = "proxy_taxi_adsb_sim"

#: (detector RMS error in seconds, fraction of rows it covers)
GRID = [(None, None),      # v4 baseline, no simulated feature
        (60, 0.90), (150, 0.90), (300, 0.90),
        (150, 0.70), (150, 0.98)]

raw = json.loads(Path("data/processed/lgbm_best_params_r3.json").read_text())
INT_P = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}
p = {k: (int(v) if k in INT_P else v) for k, v in raw.items() if not k.startswith("_")}
n_est = p.pop("n_estimators")
PARAMS = dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1, subsample_freq=1,
              random_state=0, n_estimators=n_est, cat_smooth=100.0, **p)

comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
for extra in ("train_surface", "train_config"):
    d = d.merge(pd.read_parquet(f"data/processed/{extra}.parquet"),
                on="MVT_ID_mvt", how="left", validate="one_to_one")
nums, cats = model_features(d)
cats = list(cats) + [c for c in CONFIG_CATEGORICALS if c in d.columns]
for c in cats:
    d[c] = d[c].astype("string").fillna("__NA__").astype("category")
base_feats = nums + cats

is_train, is_val, cell = mirrored_split(d)
trainable = trainable_mask(d)
taxi_true = (d["MVT_TIME_UTC_mvt"] - d["BLOCK_TIME_UTC_mvt"]).dt.total_seconds().to_numpy()
rng = np.random.default_rng(0)
vcell = cell[is_val]
print(f"{len(base_feats)} base features | fit {int((is_train & trainable).sum()):,} | "
      f"val {int(is_val.sum()):,}\n")
print(f"{'detector':<34} {'full':>8} {'trimmed':>9} {'implied board':>15} {'vs v4':>8}")
print(f"{'v4 as shipped (reference)':<34} {276.50:>8.2f} {231.49:>9.2f} {282.23:>15.2f} {'—':>8}")

for sigma, cover in GRID:
    tag = "baseline, no simulated feature" if sigma is None else \
          f"RMS {sigma:>3}s, coverage {cover:.0%}"
    dd = d
    feats = base_feats
    if sigma is not None:
        # taxi = MVT - BLOCK exactly, so a detector with error e gives taxi + e directly
        err = rng.normal(0.0, sigma, len(d))
        seen = rng.random(len(d)) < cover
        dd = d.copy()
        dd[SIM] = np.where(seen, taxi_true + err, np.nan)
        feats = base_feats + [SIM]

    tr = dd[is_train & trainable][feats + ["TAXITIME_SEC_mvt"]]
    val = dd[is_val].copy()
    regime = fit_regime_model(dd[is_train], [c for c in feats if not c.startswith(CONFIG_PREFIXES)])
    y, yv = tr["TAXITIME_SEC_mvt"], val["TAXITIME_SEC_mvt"].to_numpy()
    trimmed = trainable_mask(val)

    t0 = time.time()
    m = lgb.LGBMRegressor(**PARAMS).fit(tr[feats], y)
    pred = m.predict(val[feats])
    pred, _ = apply_regime_model(val, pred, regime)
    pred, _ = apply_extreme_proxy_override(val, pred)
    r = report(yv, pred, val, comp, vcell, trimmed_mask=trimmed)
    board = r["rmse_full"] + OFFSET
    print(f"{tag:<34} {r['rmse_full']:>8.2f} {r['rmse_trimmed']:>9.2f} {board:>15.2f} "
          f"{board-282.23:>+8.2f}   ({time.time()-t0:.0f}s)", flush=True)
    del m, tr, val, dd
    gc.collect()

print(f"\nFor reference, AOBT_3_flt's own error against the truth is 422.9s (e31).")
print("Read the rows above as: 'if our detector were this good, the board score would be that'.")
