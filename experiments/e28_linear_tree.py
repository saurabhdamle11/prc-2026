#!/usr/bin/env python
"""E28: piecewise-linear leaves, so the model can follow a proxy past the values it saw.

The failure mode E22 isolated is that a boosted tree predicts a **constant** inside each leaf,
so no matter how large `proxy_taxi_eobt1` gets, the prediction saturates at the average target
of the training rows that fell in that leaf. Measured, on normal validation rows:

    true taxi [3600, inf)   n=698   px_eobt1 median 4,014s   prediction median 2,864s

The model is not mis-calibrated there -- E22 showed an isotonic correction buys 0.01s -- it is
*saturated*. A constant leaf cannot express "keep following the proxy".

`linear_tree=True` fits a linear model in each leaf instead of a constant, which lets the
prediction track a numeric feature inside the leaf and extrapolate along it. That is exactly
the missing capability, and it is the only remaining structural change to the regressor
itself rather than to its inputs.

Compared at a **matched, reduced budget** (linear leaves cost several times more per
iteration, so a like-for-like comparison at the shipped 3,546 iterations would take hours and
confound the two effects). If it helps at 1,200 iterations it is worth a full run; if it does
not, it never will.
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

ITERS = 1200
LR = 0.06

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
vcell, trimmed = cell[is_val], trainable_mask(val)
regime = fit_regime_model(d[is_train], feats)
normal = ~(val.dq_no_aircraft_type | val.dq_implausible_target).to_numpy()

# monotone constraints: the target must not fall as any off-block proxy rises. Every one of
# these is `MVT_TIME - <an off-block estimate>`, so a larger value is a longer taxi by
# construction -- the constraint encodes an identity, not a belief about the data.
MONO = ["proxy_taxi_aobt3", "proxy_taxi_eobt1", "proxy_taxi_iobt", "proxy_taxi_lobt",
        "proxy_taxi_sched", "offblock_plan_median", "offblock_earliest", "offblock_latest"]
mono = [1 if c in MONO else 0 for c in feats]
print(f"monotone constraints on {sum(mono)} features: "
      f"{', '.join(c for c in feats if c in MONO)}\n")

BASE = dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1, subsample_freq=1,
            random_state=0, n_estimators=ITERS, learning_rate=LR, num_leaves=127,
            min_child_samples=50, colsample_bytree=0.6, subsample=0.85)


def run(label, **over):
    t0 = time.time()
    m = lgb.LGBMRegressor(**{**BASE, **over}).fit(tr[feats], y)
    pred = m.predict(val[feats])
    pred, _ = apply_regime_model(val, pred, regime)
    pred, _ = apply_extreme_proxy_override(val, pred)
    r = report(yv, pred, val, comp, vcell, trimmed_mask=trimmed)
    tail = normal & (yv >= 1800)
    print(f"{label:<38} full {r['rmse_full']:7.2f}  trimmed {r['rmse_trimmed']:7.2f}  "
          f"tail(y>=1800s, n={int(tail.sum()):,}) {np.sqrt(((pred[tail]-yv[tail])**2).mean()):7.1f}"
          f"   ({time.time()-t0:.0f}s)", flush=True)
    return r


print(f"matched budget: {ITERS} iterations at lr {LR}\n")
run("constant leaves (reference)")
run("+ monotone constraints", monotone_constraints=mono)
run("linear leaves", linear_tree=True, linear_lambda=1.0)
run("linear leaves + monotone", linear_tree=True, linear_lambda=1.0, monotone_constraints=mono)
