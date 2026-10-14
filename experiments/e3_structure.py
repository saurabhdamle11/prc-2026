#!/usr/bin/env python
"""E3: structural variants on the normal-row model.

  F1  reference            surface features, current training mask
  F2  wider training mask   also fit on implausible-target rows that HAVE an aircraft type
  F3  boost from the proxy  init_score = proxy_taxi_aobt3, so the tree fits only the residual
  F4  F2 + F3
  F5  F3 + out-of-fold target encodings of the residual

F3's motivation: `proxy_taxi_aobt3` is the dominant feature and the target is essentially
proxy + a correction. A tree can only approximate that identity by repeatedly splitting the
proxy axis, which wastes capacity and -- more importantly -- cannot extrapolate beyond the
training range, exactly where the long-taxi rows that carry a third of the error sit.
"""
import sys, time; from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness import CACHE, load, split, featurelist, encode, params, Scorer, fit_predict
from src.cleaning import trainable_mask
from src.postprocess import fit_regime_branch

d = load()
is_train, is_val, cell = split(d)
sc = Scorer(d, is_val, cell)
branch = fit_regime_branch(d[is_train])
nums, cats = featurelist(d, drop={"stand_rwy"}, extra_cats=())
d = encode(d, cats)
feats = nums + cats
print(f"{len(feats)} features")

base_mask = is_train & trainable_mask(d)
wide_mask = is_train & d["TAXITIME_SEC_mvt"].notna() & ~d["dq_no_aircraft_type"]
print(f"base fit rows {base_mask.sum():,}   wide fit rows {wide_mask.sum():,} "
      f"(+{wide_mask.sum()-base_mask.sum()} implausible-with-actype)")

val = d[is_val]
sc.val = val

def init_of(frame):
    """Offset to boost from: the ADS-B proxy, or the airport's median taxi where absent."""
    med = d[base_mask].groupby("ADEP_mvt", observed=True)["TAXITIME_SEC_mvt"].median()
    fb = frame["ADEP_mvt"].map(med).astype("float64")
    return frame["proxy_taxi_aobt3"].astype("float64").fillna(fb).fillna(float(med.median())).to_numpy()

iv = init_of(val)
res = []
for name, mask, use_init in [("F1 reference", base_mask, False),
                             ("F2 wider training mask", wide_mask, False),
                             ("F3 boost from proxy", base_mask, True),
                             ("F4 wider + boost", wide_mask, True)]:
    tr = d[mask]
    t0 = time.time()
    it = init_of(tr) if use_init else None
    _, pred = fit_predict(tr, val, feats, tr["TAXITIME_SEC_mvt"], init_tr=it,
                          init_val=iv if use_init else None)
    res.append(sc.report(pred, branch, f"{name} [{time.time()-t0:.0f}s]"))
    np.save(CACHE / f"pred_{name.split()[0]}.npy", pred)
