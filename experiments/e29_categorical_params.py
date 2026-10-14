#!/usr/bin/env python
"""E29: the categorical hyperparameters, which no sweep has ever touched.

Rounds 1 and 3 swept `learning_rate`, `num_leaves`, `min_child_samples`, `colsample_bytree`,
`subsample`, `reg_lambda`, `reg_alpha` and `max_bin`. None of them affects how LightGBM splits
a **categorical** feature, and categoricals are roughly a third of this model's total split
gain (`stand_area` 11.6%, `stand_full` 6.5%, `RUNWAY_mvt` 5.6%, `AIRCRAFT_OPERATOR_flt` 3.4%,
`ADES_mvt` 3.2%). Four parameters govern that path and all four are still at their defaults:

* `max_cat_threshold` (32) -- the most categories LightGBM will put on one side of a split.
  `stand_full` has **2,722** levels and `ADES_mvt` **1,568**, so 32 is a hard ceiling on how
  finely either can be partitioned in a single split.
* `cat_smooth` (10) and `cat_l2` (10) -- shrinkage of each category's gradient statistic
  toward the parent. With 2,722 stands over 1M rows, the rare ones are noisy and the right
  amount of shrinkage is an empirical question, not a default.
* `min_data_per_group` (100) -- the minimum rows a category needs before it gets its own
  treatment.

One-factor-at-a-time around the shipped configuration rather than a joint search: the point is
to find out whether this axis matters at all, and the protocol is only good to ±6 s (§7), so a
joint search here would just re-run round 3's mistake on different parameters.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.features import model_features
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)
from src.validation import mirrored_split, ranking_composition, report

raw = json.loads(Path("data/processed/lgbm_best_params_r3.json").read_text())
INT_P = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}
p = {k: (int(v) if k in INT_P else v) for k, v in raw.items() if not k.startswith("_")}
n_est = p.pop("n_estimators")
PARAMS = dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1,
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
print("categorical cardinalities: " +
      ", ".join(f"{c}={d[c].nunique()}" for c in cats) + "\n")

is_train, is_val, cell = mirrored_split(d)
tr, val = d[is_train & trainable_mask(d)], d[is_val].copy()
y, yv = tr["TAXITIME_SEC_mvt"], val["TAXITIME_SEC_mvt"].to_numpy()
vcell, trimmed = cell[is_val], trainable_mask(val)
regime = fit_regime_model(d[is_train], feats)


def run(label, **over):
    t0 = time.time()
    m = lgb.LGBMRegressor(**{**PARAMS, **over}).fit(tr[feats], y)
    pred = m.predict(val[feats])
    pred, _ = apply_regime_model(val, pred, regime)
    pred, _ = apply_extreme_proxy_override(val, pred)
    r = report(yv, pred, val, comp, vcell, trimmed_mask=trimmed)
    print(f"{label:<40} full {r['rmse_full']:7.2f}  trimmed {r['rmse_trimmed']:7.2f}"
          f"   ({time.time()-t0:.0f}s)", flush=True)
    return r


# `--cat-smooth` re-runs just that axis. The first pass found `cat_smooth` monotone
# (1.0 -> 280.47, 10 -> 279.82, 50 -> 279.35) while every other categorical parameter was
# worse in both directions, so the follow-up looks for where the trend turns rather than
# stopping at the edge of an arbitrary grid.
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--cat-smooth", type=float, nargs="*", default=None)
a = ap.parse_args()

if a.cat_smooth is not None:
    run("defaults (cat_smooth=10)")
    for v in a.cat_smooth:
        run(f"cat_smooth={v}", cat_smooth=v)
else:
    run("defaults (shipped categorical handling)")
    for v in (64, 128, 512):
        run(f"max_cat_threshold={v}", max_cat_threshold=v)
    for v in (1.0, 50.0):
        run(f"cat_smooth={v}", cat_smooth=v)
    for v in (1.0, 50.0):
        run(f"cat_l2={v}", cat_l2=v)
    for v in (20, 500):
        run(f"min_data_per_group={v}", min_data_per_group=v)
