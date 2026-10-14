#!/usr/bin/env python
"""E30: the exact configuration v4 ships, measured end to end on the mirrored protocol.

Every round-4 change that survived, measured together rather than summed from separate
experiments — the repo's own rule is that what gets quoted is what gets shipped:

* the 12 runway-configuration and stand-rotation features (`src/config_features.py`), -1.36 s
* the round-3 hyperparameters (`lgbm_best_params_r3.json`), -1.67 s
* `cat_smooth`, the one categorical parameter that moved and did so monotonically
* 3-seed bagging, -0.84 s
* both shipped post-processing rules, unchanged

Deliberately **not** included: the 17 off-block gap features from `e23`. They measured
-0.52 s, which is a tenth of the protocol's precision, and wiring them into the production
path means touching `src/cleaning.py` and both feature builds for a change that cannot be
distinguished from noise. They remain in `e23` if a later round wants them.

    python experiments/e30_final_r4.py --cat-smooth 50
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.config_features import CONFIG_CATEGORICALS
from src.features import model_features
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)
from src.validation import mirrored_split, ranking_composition, report

CONFIG_PREFIXES = ("depcfg_", "arrcfg_", "stand_turnaround", "stand_prev_arr")

ap = argparse.ArgumentParser()
ap.add_argument("--cat-smooth", type=float, default=50.0)
ap.add_argument("--seeds", type=int, default=3)
args = ap.parse_args()

raw = json.loads(Path("data/processed/lgbm_best_params_r3.json").read_text())
INT_P = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}
p = {k: (int(v) if k in INT_P else v) for k, v in raw.items() if not k.startswith("_")}
n_est = p.pop("n_estimators")
PARAMS = dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1,
              subsample_freq=1, n_estimators=n_est, cat_smooth=args.cat_smooth, **p)

comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
for extra in ("train_surface", "train_config"):
    n0 = len(d)
    d = d.merge(pd.read_parquet(f"data/processed/{extra}.parquet"),
                on="MVT_ID_mvt", how="left", validate="one_to_one")
    assert len(d) == n0, f"{extra} join changed the row count"

nums, cats = model_features(d)
cats = list(cats) + [c for c in CONFIG_CATEGORICALS if c in d.columns]
for c in cats:
    d[c] = d[c].astype("string").fillna("__NA__").astype("category")
feats = nums + cats

is_train, is_val, cell = mirrored_split(d)
tr = d[is_train & trainable_mask(d)][feats + ["TAXITIME_SEC_mvt"]].copy()
val = d[is_val].copy()
regime = fit_regime_model(d[is_train], [c for c in feats if not c.startswith(CONFIG_PREFIXES)])
vcell, trimmed = cell[is_val], trainable_mask(val)
del d, is_train, is_val, cell
gc.collect()

y = tr.pop("TAXITIME_SEC_mvt")
yv = val["TAXITIME_SEC_mvt"].to_numpy()
print(f"{len(feats)} features | fit {len(tr):,} | val {len(val):,} | {n_est} iters | "
      f"cat_smooth={args.cat_smooth} | {args.seeds} seeds", flush=True)

t0, preds = time.time(), []
for s in range(args.seeds):
    m = lgb.LGBMRegressor(**PARAMS, random_state=s).fit(tr[feats], y)
    preds.append(m.predict(val[feats]))
    del m; gc.collect()
    print(f"  seed {s} ({time.time()-t0:.0f}s)", flush=True)

pred = np.mean(preds, axis=0)
r_raw = report(yv, pred, val, comp, vcell, trimmed_mask=trimmed)
pred, n1 = apply_regime_model(val, pred, regime)
pred, n2 = apply_extreme_proxy_override(val, pred)
r = report(yv, pred, val, comp, vcell, trimmed_mask=trimmed)

print(f"\n  before post-processing   full {r_raw['rmse_full']:7.2f}  trimmed {r_raw['rmse_trimmed']:7.2f}")
print(f"  + regime model ({n1} rows) + override ({n2} rows)")
print(f"\n*** v4 CANDIDATE   full {r['rmse_full']:7.2f}   trimmed {r['rmse_trimmed']:7.2f} ***")
print(f"    v3 as shipped  full  281.49   trimmed  237.30   (board 287.10)")
print(f"    delta          full {r['rmse_full']-281.49:+7.2f}   trimmed {r['rmse_trimmed']-237.30:+7.2f}")
np.save("data/processed/exp_cache/r4_pred_final.npy", pred)
