#!/usr/bin/env python
"""E33: does the E32 result survive a detector that fails the way AOBT_3 fails?

E32 says a detector at 150 s RMS with 90% coverage would score ~165 on the board against v4's
282. That result rests on an assumption the table does not show: **the error is Gaussian and
independent of the target**. `AOBT_3_flt`'s error is emphatically not -- it is -67 s on
ordinary taxis and **+2,758 s** on taxis over an hour, concentrated exactly where the squared
error lives. A detector that inherits that pathology would look fine on headline RMS and be
worth far less.

So this re-runs the simulation with **heteroscedastic error**: small on ordinary taxis, large
on the long ones, i.e. a detector that still misses apron holds. If the project survives that,
the investment case does not depend on the detector being well behaved in the tail.

The original sizing argument, for reference: Training rows carry `BLOCK_TIME_UTC_mvt`, so a detector of any
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

#: (label, sigma on ordinary taxis, sigma on taxis >= 1800s, coverage)
#: The last two mimic a detector that resolves normal pushbacks well but still misses holds --
#: the failure mode `AOBT_3_flt` already has.
GRID = [("baseline, no simulated feature", None, None, None),
        ("flat 150s (E32 assumption)",      150,  150, 0.90),
        ("150s normal / 800s on holds",     150,  800, 0.90),
        ("150s normal / 2000s on holds",    150, 2000, 0.90),
        ("150s normal / 2000s, 70% cover",  150, 2000, 0.70)]

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

is_hold = taxi_true >= 1800      # the population AOBT_3 mishandles
for tag, s_norm, s_hold, cover in GRID:
    dd = d
    feats = base_feats
    if s_norm is not None:
        sig = np.where(is_hold, s_hold, s_norm).astype(float)
        err = rng.normal(0.0, 1.0, len(d)) * sig
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

print(f"\nAOBT_3_flt's own error is 422.9s overall, but +2,758s median on taxis over an hour.")
print("If the degraded rows still beat 224, the case does not rest on tail behaviour.")
