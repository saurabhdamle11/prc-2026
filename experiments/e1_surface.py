#!/usr/bin/env python
"""E1: do the rolling surface-state features and the stand x runway interaction help?"""
import sys, time; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness import *  # noqa
from src.cleaning import trainable_mask
from src.postprocess import fit_regime_branch

d = load()
surf = [c for c in d.columns if c.startswith("surf_")]
print(f"{len(d):,} rows, {len(surf)} surface features")
print(d[surf].notna().mean().round(3).to_string())

is_train, is_val, cell = split(d)
sc = Scorer(d, is_val, cell)
branch = fit_regime_branch(d[is_train])
tr_mask = is_train & trainable_mask(d)

for name, drop, xc in [("baseline (88 feat)", set(surf) | {"stand_rwy"}, ()),
                       ("+ stand_rwy",        set(surf),                ("stand_rwy",)),
                       ("+ surface state",    {"stand_rwy"},            ()),
                       ("+ both",             set(),                    ("stand_rwy",))]:
    nums, cats = featurelist(d, drop=drop, extra_cats=xc)
    dd = encode(d.copy(), cats)
    feats = nums + cats
    tr, val = dd[tr_mask], dd[is_val]
    t0 = time.time()
    m, pred = fit_predict(tr, val, feats, tr["TAXITIME_SEC_mvt"])
    sc.val = val
    sc.report(pred, branch, f"{name} [{len(feats)}f, {time.time()-t0:.0f}s]")
