#!/usr/bin/env python
"""E27: do runway configuration and stand rotation move the bulk model?

The two families `e18_gaps.py` identified in round 2 and that were never built or recorded.
Built by `e20_config.py`; this measures them through the mirrored protocol.

The case for them, from E19/E21: the error is concentrated in the *normal* rows now (70.6% of
the weighted MSE), and the shipped features describe how much traffic surrounds a flight and
how slowly the surface is running, but never **which physical layout the airport is working**.
Two departures from the same stand to the same runway travel different distances depending on
which other runways are active, and that is invisible to the current 127.

Also tests seed bagging -- averaging fits that differ only in their random seed, which reduces
variance without touching bias.

**One arm per process.** The first version of this script ran all arms in one interpreter and
was OOM-killed on a 16 GB machine partway through the second: the feature frame is ~2.1M rows
wide, and `tr[feats]` materialises another copy per fit. Each arm is now its own run, so the
frame is freed between them. Drive it with::

    for a in base config bag; do python experiments/e27_config_model.py --arm $a; done
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.features import model_features
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)
from src.validation import mirrored_split, ranking_composition, report

CFG_CATS = ["depcfg_cfg_id", "arrcfg_cfg_id"]
NEW_PREFIXES = ("depcfg_", "arrcfg_", "stand_turnaround", "stand_prev_arr")

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["base", "config", "bag", "bag-base"], required=True)
ap.add_argument("--seeds", type=int, default=3, help="fits to average in the bagging arms")
args = ap.parse_args()

raw = json.loads(Path("data/processed/lgbm_best_params_r3.json").read_text())
INT_P = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}
p = {k: (int(v) if k in INT_P else v) for k, v in raw.items() if not k.startswith("_")}
n_est = p.pop("n_estimators")
PARAMS = dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1,
              subsample_freq=1, n_estimators=n_est, **p)

comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
d = d.merge(pd.read_parquet("data/processed/train_surface.parquet"),
            on="MVT_ID_mvt", how="left", validate="one_to_one")
want_new = args.arm in ("config", "bag")
if want_new:
    n0 = len(d)
    d = d.merge(pd.read_parquet("data/processed/train_config.parquet"),
                on="MVT_ID_mvt", how="left", validate="one_to_one")
    assert len(d) == n0, "config join changed the row count"

nums, cats = model_features(d)
cats = list(cats) + [c for c in CFG_CATS if c in d.columns]
for c in cats:
    d[c] = d[c].astype("string").fillna("__NA__").astype("category")
feats = [c for c in nums + cats
         if want_new or not c.startswith(NEW_PREFIXES)]
new = [c for c in feats if c.startswith(NEW_PREFIXES)]

is_train, is_val, cell = mirrored_split(d)
tr = d[is_train & trainable_mask(d)][feats + ["TAXITIME_SEC_mvt"]].copy()
val = d[is_val].copy()
regime = fit_regime_model(d[is_train], [c for c in feats if not c.startswith(NEW_PREFIXES)])
vcell, trimmed = cell[is_val], trainable_mask(val)
del d, is_train, is_val, cell
gc.collect()

y = tr.pop("TAXITIME_SEC_mvt")
yv = val["TAXITIME_SEC_mvt"].to_numpy()
seeds = range(args.seeds) if args.arm.startswith("bag") else (0,)
print(f"arm={args.arm} | {len(feats)} features ({len(new)} new) | fit {len(tr):,} | "
      f"val {len(val):,} | {n_est} iters | seeds {list(seeds)}", flush=True)

t0, preds = time.time(), []
for s in seeds:
    m = lgb.LGBMRegressor(**PARAMS, random_state=s).fit(tr[feats], y)
    preds.append(m.predict(val[feats]))
    del m
    gc.collect()
    print(f"  seed {s} done ({time.time()-t0:.0f}s)", flush=True)

pred = np.mean(preds, axis=0)
pred, _ = apply_regime_model(val, pred, regime)
pred, _ = apply_extreme_proxy_override(val, pred)
r = report(yv, pred, val, comp, vcell, trimmed_mask=trimmed)
print(f"\nRESULT arm={args.arm:<10} full {r['rmse_full']:7.2f}  trimmed {r['rmse_trimmed']:7.2f}"
      f"   ({time.time()-t0:.0f}s total)")
np.save(f"data/processed/exp_cache/r4_pred_{args.arm}.npy", pred)
