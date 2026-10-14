#!/usr/bin/env python
"""E36: do the OpenStreetMap taxi-geometry features move the model?

`e35` established the joins land — 88.1% of departures matched to a mapped stand, 100% of
runway thresholds, and **100% at LIRF**, which alone carries 42% of the weighted MSE and where
the abandoned ADS-B route had no coverage at all. The within-airport residual correlation is
+0.210, so there is signal. This measures whether the model can use it.

The prior is modest. `SKILLS.md` §8 records that `stand_area x RUNWAY_mvt` as an explicit
categorical changed nothing, because LightGBM recovers that interaction wherever the data is
plentiful. What a numeric distance adds is **generalisation to the rare stand-runway pairs** —
real, but bounded. One seed per arm: the comparison is relative and this halves the cost.
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.config_features import CONFIG_CATEGORICALS
from src.features import model_features
from src.osm_features import OSM_FEATURES
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)
from src.validation import mirrored_split, ranking_composition, report

OFFSET = 5.73
CONFIG_PREFIXES = ("depcfg_", "arrcfg_", "stand_turnaround", "stand_prev_arr")

raw = json.loads(Path("data/processed/lgbm_best_params_r3.json").read_text())
INT_P = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}
p = {k: (int(v) if k in INT_P else v) for k, v in raw.items() if not k.startswith("_")}
n_est = p.pop("n_estimators")
PARAMS = dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1, subsample_freq=1,
              random_state=0, n_estimators=n_est, cat_smooth=100.0, **p)

comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
for extra in ("train_surface", "train_config", "train_osm"):
    d = d.merge(pd.read_parquet(f"data/processed/{extra}.parquet"),
                on="MVT_ID_mvt", how="left", validate="one_to_one")
nums, cats = model_features(d)
cats = list(cats) + [c for c in CONFIG_CATEGORICALS if c in d.columns]
for c in cats:
    d[c] = d[c].astype("string").fillna("__NA__").astype("category")
all_feats = nums + cats
base_feats = [c for c in all_feats if c not in set(OSM_FEATURES)]

is_train, is_val, cell = mirrored_split(d)
trainable = trainable_mask(d)
tr_all = d[is_train & trainable]
val = d[is_val].copy()
regime = fit_regime_model(d[is_train], [c for c in base_feats if not c.startswith(CONFIG_PREFIXES)])
vcell, trimmed = cell[is_val], trainable_mask(val)
yv = val["TAXITIME_SEC_mvt"].to_numpy()
y = tr_all["TAXITIME_SEC_mvt"]
normal = ~(val.dq_no_aircraft_type | val.dq_implausible_target).to_numpy()
is_lirf = (val.ADEP_mvt.astype(str) == "LIRF").to_numpy()
print(f"fit {len(tr_all):,} | val {len(val):,} | base {len(base_feats)} feats "
      f"| +{len(OSM_FEATURES)} OSM\n")
print(f"{'arm':<28} {'full':>8} {'trimmed':>9} {'LIRF rmse':>11} {'implied board':>15}")


def run(feats, label):
    t0 = time.time()
    m = lgb.LGBMRegressor(**PARAMS).fit(tr_all[feats], y)
    pred = m.predict(val[feats])
    pred, _ = apply_regime_model(val, pred, regime)
    pred, _ = apply_extreme_proxy_override(val, pred)
    r = report(yv, pred, val, comp, vcell, trimmed_mask=trimmed)
    lirf = np.sqrt(((pred[is_lirf & normal] - yv[is_lirf & normal]) ** 2).mean())
    print(f"{label:<28} {r['rmse_full']:>8.2f} {r['rmse_trimmed']:>9.2f} {lirf:>11.1f} "
          f"{r['rmse_full']+OFFSET:>15.2f}   ({time.time()-t0:.0f}s)", flush=True)
    del m; gc.collect()
    return r


r0 = run(base_feats, "v4 baseline (1 seed)")
r1 = run(base_feats + OSM_FEATURES, f"+ {len(OSM_FEATURES)} OSM geometry")
print(f"\ndelta: {r1['rmse_full']-r0['rmse_full']:+.2f}s full   "
      f"{r1['rmse_trimmed']-r0['rmse_trimmed']:+.2f}s trimmed")
