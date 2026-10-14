#!/usr/bin/env python
"""E7: finer-grained realised-taxi features (per runway, per apron, per apron->runway pair).

The airport-level surface features helped (242.8 -> 238.3 trimmed). Counts never did. The
hypothesis under test is that what carries the signal is *realised duration*, so measuring it
at the granularity the aircraft actually travels -- this apron, to this runway -- should carry
more of it than an airport-wide average that mixes a 400 m pushback with a 3 km taxi.
"""
import sys, time; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness import CACHE, load, split, featurelist, encode, Scorer, fit_predict
from src.cleaning import trainable_mask
from src.postprocess import fit_regime_branch

d = load()
keyed = [c for c in d.columns if c.startswith(("surf_rwy_", "surf_area_", "surf_arearwy_"))]
apt_lvl = [c for c in d.columns if c.startswith("surf_") and c not in keyed]
print(f"{len(apt_lvl)} airport-level + {len(keyed)} keyed surface features")
print(d[keyed].notna().mean().round(3).to_string())

is_train, is_val, cell = split(d)
sc = Scorer(d, is_val, cell)
branch = fit_regime_branch(d[is_train])
tr_mask = is_train & trainable_mask(d)

for name, drop in [("G1 airport-level only", set(keyed)),
                   ("G2 + keyed surface", set())]:
    nums, cats = featurelist(d, drop=drop | {"stand_rwy"}, extra_cats=())
    dd = encode(d.copy(), cats)
    feats = nums + cats
    tr, val = dd[tr_mask], dd[is_val]
    sc.val = val
    t0 = time.time()
    _, pred = fit_predict(tr, val, feats, tr["TAXITIME_SEC_mvt"])
    sc.report(pred, branch, f"{name} [{len(feats)}f, {time.time()-t0:.0f}s]")
    np.save(CACHE / f"pred_{name.split()[0]}.npy", pred)
