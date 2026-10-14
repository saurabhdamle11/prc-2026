#!/usr/bin/env python
"""E14: does blending CatBoost with LightGBM beat either alone on the final feature set?

The two libraries handle high-cardinality categoricals very differently (ordered target
statistics vs gradient-sorted splits), so their errors are only partly correlated -- which is
the condition under which a blend helps. Cross-checked on the identical mirrored split.
"""
import sys, time; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness import CACHE, load, split, featurelist, encode, params, Scorer, fit_predict
from src.cleaning import trainable_mask
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)

S = CACHE
d = load()
is_train, is_val, cell = split(d)
sc = Scorer(d, is_val, cell)
nums, cats = featurelist(d, drop={"stand_rwy"}, extra_cats=())
d = encode(d, cats)
feats = nums + cats
tr, val = d[is_train & trainable_mask(d)], d[is_val]
sc.val = val
y = tr["TAXITIME_SEC_mvt"]
print(f"{len(feats)} features | fit {len(tr):,} | val {len(val):,}", flush=True)

pl = S/"final_p_surface_p_corroborate.npy"
if pl.exists():
    lgbm_pred = np.load(pl); print("reusing LightGBM predictions from e13")
else:
    _, lgbm_pred = fit_predict(tr, val, feats, y); np.save(pl, lgbm_pred)

import catboost as cb
cat_idx = [feats.index(c) for c in cats]
tr_c, val_c = tr[feats].copy(), val[feats].copy()
for c in cats:
    tr_c[c] = tr_c[c].astype(str); val_c[c] = val_c[c].astype(str)
t0 = time.time()
m = cb.CatBoostRegressor(iterations=2000, depth=8, learning_rate=0.06, loss_function="RMSE",
                         random_seed=0, verbose=500, thread_count=-1)
m.fit(tr_c, y, cat_features=cat_idx)
cat_pred = m.predict(val_c)
np.save(S/"cat_pred.npy", cat_pred)
print(f"catboost fitted in {time.time()-t0:.0f}s", flush=True)

branch = fit_regime_model(d[is_train], feats)
def full(p):
    p1, _ = apply_regime_model(val, p, branch)
    p2, _ = apply_extreme_proxy_override(val, p1)
    return sc._w(p2), sc._w(p2, sc.trim)

print(f"\n{'blend w(lgbm)':<16}{'full':>9}{'trimmed':>10}")
for w in [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]:
    f, t = full(w*lgbm_pred + (1-w)*cat_pred)
    print(f"{w:<16.2f}{f:>9.1f}{t:>10.1f}")
