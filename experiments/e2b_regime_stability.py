#!/usr/bin/env python
"""E2b: is the delta-target regime model a real gain or a 428-row fluke?

Repeats the comparison across several disjoint train/holdout month splits and several seeds,
scoring the regime rows on their own (plain RMSE over the regime population), so the answer
does not depend on how the rest of the model happens to be doing.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness import load, featurelist, encode

d = load()
nums, cats = featurelist(d, drop={"stand_rwy"}, extra_cats=())
d = encode(d, cats)
feats = nums + cats
d["_m"] = d["MVT_TIME_UTC_mvt"].dt.month
R = d[d["dq_no_aircraft_type"] & d["TAXITIME_SEC_mvt"].notna()].copy()
usable = [c for c in feats if R[c].notna().mean() > 0.5]
print(f"{len(R)} regime rows total, {len(usable)} features populated on them\n")

def rmse(p, y): return float(np.sqrt(np.mean((np.maximum(p,0)-y)**2)))

splits = [([1,2,3,4,5,6], [7,8,9]), ([1,2,3,4,5,6], [12]), ([1,2,3,4,5,6], [7,8,9,10,11,12]),
          ([1,2,3,4,5,6,7,8,9], [10,11,12]), ([4,5,6,7,8,9], [1,2,3]), ([7,8,9,10,11,12], [1,2,3,4,5,6])]
rows = []
for tm, vm in splits:
    tr, va = R[R._m.isin(tm)], R[R._m.isin(vm)]
    if len(tr) < 100 or len(va) < 50: continue
    y = va["TAXITIME_SEC_mvt"].to_numpy()
    const = rmse(np.full(len(va), tr.TAXITIME_SEC_mvt.mean()), y)
    ok = tr["proxy_taxi_sched"].notna()
    s, i = np.polyfit(tr.loc[ok,"proxy_taxi_sched"], tr.loc[ok,"TAXITIME_SEC_mvt"], 1)
    lin = rmse(np.nan_to_num(s*va["proxy_taxi_sched"].to_numpy()+i, nan=tr.TAXITIME_SEC_mvt.mean()), y)
    dl = []
    for seed in range(5):
        tgt = tr["proxy_taxi_sched"] - tr["TAXITIME_SEC_mvt"]; k = tgt.notna()
        m = lgb.LGBMRegressor(objective="regression", num_leaves=8, min_child_samples=20,
                              n_estimators=300, learning_rate=0.05, colsample_bytree=0.7,
                              subsample=0.8, subsample_freq=1, verbose=-1, random_state=seed)
        m.fit(tr.loc[k, usable], tgt[k])
        v = va["proxy_taxi_sched"].to_numpy() - m.predict(va[usable])
        dl.append(rmse(np.where(np.isfinite(v), v, tr.TAXITIME_SEC_mvt.mean()), y))
    rows.append(dict(train=str(tm), val=str(vm), n_tr=len(tr), n_va=len(va),
                     const=const, linear=lin, delta_mean=np.mean(dl), delta_sd=np.std(dl)))
r = pd.DataFrame(rows)
r["delta_vs_linear_%"] = 100*(r.delta_mean-r.linear)/r.linear
print(r.round(1).to_string(index=False))
print(f"\ndelta beats linear on {(r.delta_mean < r.linear).sum()}/{len(r)} splits; "
      f"mean relative change {r['delta_vs_linear_%'].mean():.1f}%")
