#!/usr/bin/env python
"""E38: the exact v5 configuration — v4 plus the two sub-noise feature families, together.

Both were measured alone against v4 and both landed inside the protocol's ±6 s:
off-block gaps −0.52 s (`e23`), OSM taxi geometry −0.44 s (`e36`). Neither is a result on its
own. The question here is only whether they are additive and non-negative together, because
the repo's rule is that what gets quoted is what gets shipped.

Neither adds tail risk: both are pure feature additions, the post-processing rules are
untouched, and v4's 2-row extreme-proxy bet is carried unchanged.
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
              n_estimators=n_est, cat_smooth=100.0, **p)

comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
for extra in ("train_surface", "train_config", "train_osm"):
    d = d.merge(pd.read_parquet(f"data/processed/{extra}.parquet"),
                on="MVT_ID_mvt", how="left", validate="one_to_one")
d, gaps = add_offblock_gaps(d)
nums, cats = model_features(d)
cats = list(cats) + [c for c in CONFIG_CATEGORICALS if c in d.columns]
for c in cats:
    d[c] = d[c].astype("string").fillna("__NA__").astype("category")
all_feats = nums + cats
extra_set = set(gaps) | set(OSM_FEATURES)
v4_feats = [c for c in all_feats if c not in extra_set]

is_train, is_val, cell = mirrored_split(d)
trainable = trainable_mask(d)
tr = d[is_train & trainable]
val = d[is_val].copy()
regime = fit_regime_model(d[is_train], [c for c in v4_feats if not c.startswith(CONFIG_PREFIXES)])
vcell, trimmed = cell[is_val], trainable_mask(val)
y, yv = tr["TAXITIME_SEC_mvt"], val["TAXITIME_SEC_mvt"].to_numpy()
print(f"v4 {len(v4_feats)} feats | +{len(gaps)} gaps +{len(OSM_FEATURES)} OSM "
      f"= {len(all_feats)} | fit {len(tr):,}\n")


def run(feats, label, seeds):
    t0, preds = time.time(), []
    for s in seeds:
        m = lgb.LGBMRegressor(**PARAMS, random_state=s).fit(tr[feats], y)
        preds.append(m.predict(val[feats])); del m; gc.collect()
    pred = np.mean(preds, axis=0)
    pred, _ = apply_regime_model(val, pred, regime)
    pred, _ = apply_extreme_proxy_override(val, pred)
    r = report(yv, pred, val, comp, vcell, trimmed_mask=trimmed)
    print(f"{label:<34} full {r['rmse_full']:>7.2f}  trimmed {r['rmse_trimmed']:>7.2f}  "
          f"board {r['rmse_full']+OFFSET:>7.2f}   ({time.time()-t0:.0f}s)", flush=True)
    return r


print("single seed, for a like-for-like comparison:")
a = run(v4_feats, "v4 features", (0,))
b = run(all_feats, "v5 = v4 + gaps + OSM", (0,))
print(f"\ndelta: {b['rmse_full']-a['rmse_full']:+.2f}s full  "
      f"{b['rmse_trimmed']-a['rmse_trimmed']:+.2f}s trimmed")
if b["rmse_trimmed"] < a["rmse_trimmed"]:
    print("\n3-seed bag of the v5 configuration (what would ship):")
    run(all_feats, "v5, 3-seed bag", (0, 1, 2))
else:
    print("\nv5 is not better on the trimmed number — not shipping it.")
