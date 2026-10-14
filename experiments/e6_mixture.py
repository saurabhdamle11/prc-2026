#!/usr/bin/env python
"""E6: a two-component mixture for the missing-aircraft-type regime.

The regime is two populations, not one:

  * ~48% have the block time stamped within seconds of the scheduled time, so
    ``y == proxy_taxi_sched`` and the answer is *known*, not estimated;
  * ~52% are ordinary taxis whose block time is unrelated to the schedule.

Which mode a row is in is predictable from fields present in ranking.parquet -- stand_area
separates it from 5% to 64%, proxy_taxi_sched from 28% to 67%. So the MSE-optimal prediction
is the mixture mean

    E[y|x] = p(x) * proxy_taxi_sched  +  (1 - p(x)) * E[y | ordinary, x]

rather than a single line fitted through both modes, which is wrong for each of them.

Scored across six disjoint month splits so the choice does not rest on one holdout.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness import load, featurelist, encode

TOL = 60          # |SCHED - BLOCK| under which the schedule *is* the block time
d = load()
nums, cats = featurelist(d, drop={"stand_rwy"}, extra_cats=())
d = encode(d, cats)
feats = nums + cats
d["_m"] = d["MVT_TIME_UTC_mvt"].dt.month
R = d[d["dq_no_aircraft_type"] & d["TAXITIME_SEC_mvt"].notna()].copy()
R["_mode"] = ((R["TAXITIME_SEC_mvt"] - R["proxy_taxi_sched"]).abs() <= TOL).astype(int)
U = [c for c in feats if R[c].notna().mean() > 0.5]
print(f"{len(R)} regime rows, {len(U)} usable features, mode rate {R._mode.mean():.1%}\n")

def rmse(p, y): return float(np.sqrt(np.mean((np.maximum(np.asarray(p, float), 0) - y) ** 2)))

def fit_linear(tr):
    ok = tr["proxy_taxi_sched"].notna()
    return np.polyfit(tr.loc[ok, "proxy_taxi_sched"], tr.loc[ok, "TAXITIME_SEC_mvt"], 1)

def fit_delta(tr, seed=0):
    t = tr["proxy_taxi_sched"] - tr["TAXITIME_SEC_mvt"]; k = t.notna()
    m = lgb.LGBMRegressor(objective="regression", num_leaves=8, min_child_samples=20,
                          n_estimators=300, learning_rate=0.05, colsample_bytree=0.7,
                          subsample=0.8, subsample_freq=1, verbose=-1, random_state=seed)
    m.fit(tr.loc[k, U], t[k]); return m

def fit_mixture(tr, seed=0, leaves=8, mcs=20):
    clf = lgb.LGBMClassifier(objective="binary", num_leaves=leaves, min_child_samples=mcs,
                             n_estimators=250, learning_rate=0.05, colsample_bytree=0.7,
                             subsample=0.8, subsample_freq=1, verbose=-1, random_state=seed)
    clf.fit(tr[U], tr["_mode"])
    o = tr[tr["_mode"] == 0]
    reg = lgb.LGBMRegressor(objective="regression", num_leaves=leaves, min_child_samples=mcs,
                            n_estimators=250, learning_rate=0.05, colsample_bytree=0.7,
                            subsample=0.8, subsample_freq=1, verbose=-1, random_state=seed)
    reg.fit(o[U], o["TAXITIME_SEC_mvt"])
    return clf, reg, float(o["TAXITIME_SEC_mvt"].mean()), float(tr["TAXITIME_SEC_mvt"].mean())

def pred_mixture(mdl, va):
    clf, reg, ord_mean, all_mean = mdl
    p = clf.predict_proba(va[U])[:, 1]
    ps = va["proxy_taxi_sched"].to_numpy(float)
    ordp = reg.predict(va[U])
    out = p * ps + (1 - p) * ordp
    return np.where(np.isfinite(ps), out, np.where(np.isfinite(ordp), ordp, all_mean))

splits = [([1,2,3,4,5,6],[7,8,9]), ([1,2,3,4,5,6],[12]), ([1,2,3,4,5,6],[7,8,9,10,11,12]),
          ([1,2,3,4,5,6,7,8,9],[10,11,12]), ([4,5,6,7,8,9],[1,2,3]), ([7,8,9,10,11,12],[1,2,3,4,5,6])]
rows = []
for tm, vm in splits:
    tr, va = R[R._m.isin(tm)], R[R._m.isin(vm)]
    y = va["TAXITIME_SEC_mvt"].to_numpy()
    s, i = fit_linear(tr)
    r = {"val": str(vm), "n_va": len(va),
         "linear": rmse(np.nan_to_num(s*va.proxy_taxi_sched.to_numpy()+i, nan=tr.TAXITIME_SEC_mvt.mean()), y)}
    r["delta"] = np.mean([rmse(np.nan_to_num(va.proxy_taxi_sched.to_numpy()-fit_delta(tr,sd).predict(va[U]),
                                             nan=tr.TAXITIME_SEC_mvt.mean()), y) for sd in range(3)])
    r["mixture"] = np.mean([rmse(pred_mixture(fit_mixture(tr, sd), va), y) for sd in range(3)])
    # oracle on the mode only: how much of the gap is the classifier, how much the regressor
    mdl = fit_mixture(tr, 0); _, reg, ordm, allm = mdl
    ps, op = va.proxy_taxi_sched.to_numpy(float), reg.predict(va[U])
    om = va["_mode"].to_numpy()
    r["mode-oracle"] = rmse(np.where(om & np.isfinite(ps), ps, op), y)
    rows.append(r)
t = pd.DataFrame(rows)
print(t.round(0).to_string(index=False))
print("\nmean over splits:")
print(t[["linear","delta","mixture","mode-oracle"]].mean().round(0).to_string())
print(f"\nmixture beats linear on {(t.mixture<t.linear).sum()}/{len(t)} splits, "
      f"delta on {(t.mixture<t['delta']).sum()}/{len(t)}")
