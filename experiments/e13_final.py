#!/usr/bin/env python
"""E13: the consolidated pipeline -- every change that survived its own test, measured together.

  base            88 original features
  + surface       rolling realised-taxi state (airport level + keyed)
  + corroborate   agreement among the independent off-block estimates
  post-processing: delta regime model, then the corroborated-extreme-proxy override
"""
import sys, time; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness import CACHE, load, split, featurelist, encode, Scorer, fit_predict
from src.cleaning import trainable_mask
from src.postprocess import (apply_extreme_proxy_override, apply_regime_branch,
                             apply_regime_model, fit_regime_branch, fit_regime_model)
from src.validation import weighted_rmse

d = load()
CORR = [c for c in ["offblock_agree_n","offblock_agree_min","offblock_plan_median",
                    "offblock_plan_spread","proxy_vs_plan"] if c in d.columns]
SURF = [c for c in d.columns if c.startswith("surf_")]
print(f"surface {len(SURF)}  corroboration {len(CORR)}: {CORR}")

is_train, is_val, cell = split(d)
sc = Scorer(d, is_val, cell)
tr_mask = is_train & trainable_mask(d)

for name, drop in [("base", set(SURF) | set(CORR)),
                   ("+ surface", set(CORR)),
                   ("+ surface + corroborate", set())]:
    nums, cats = featurelist(d, drop=drop | {"stand_rwy"}, extra_cats=())
    dd = encode(d.copy(), cats)
    feats = nums + cats
    tr, val = dd[tr_mask], dd[is_val]
    sc.val = val
    t0 = time.time()
    _, pred = fit_predict(tr, val, feats, tr["TAXITIME_SEC_mvt"])
    y, vc, va = sc.y, sc.cell, sc.apt
    lin = fit_regime_branch(dd[is_train])
    dlt = fit_regime_model(dd[is_train], feats)
    p_lin, _ = apply_regime_branch(val, pred, lin)
    p_dlt, _ = apply_regime_model(val, pred, dlt)
    p_fin, n_ov = apply_extreme_proxy_override(val, p_dlt)
    W = lambda p: weighted_rmse(y, p, vc, va, comp) if False else sc._w(p)
    print(f"\n{name}  [{len(feats)}f, {time.time()-t0:.0f}s]   trimmed {sc._w(pred, sc.trim):7.1f}")
    print(f"   raw                              {sc._w(pred):7.1f}")
    print(f"   + linear regime branch (shipped) {sc._w(p_lin):7.1f}")
    print(f"   + delta regime model             {sc._w(p_dlt):7.1f}")
    print(f"   + extreme-proxy override ({n_ov} rows) {sc._w(p_fin):7.1f}", flush=True)
    np.save(CACHE / f"final_{name.replace(' ','_').replace('+','p')}.npy", pred)
