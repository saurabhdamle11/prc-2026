#!/usr/bin/env python
"""E2: a real model for the missing-aircraft-type regime, replacing the 1-D linear rule.

Scored by plugging each candidate into the cached baseline prediction vector, so no refit of
the main model is needed and every variant is compared on identical normal-row predictions.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import HuberRegressor, Ridge
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.validation import mirrored_split, ranking_composition, weighted_rmse
from src.postprocess import fit_regime_branch, apply_regime_branch
from experiments.harness import CACHE, load, featurelist, encode, params


d = load()
is_train, is_val, cell = mirrored_split(d)
comp = ranking_composition(pd.read_parquet("Dataset/ranking.parquet",
        columns=["PHASE_mvt","MVT_TIME_UTC_mvt","ADEP_mvt"]))
val = d[is_val].copy(); y = val["TAXITIME_SEC_mvt"].to_numpy()
base = np.load(CACHE/"base_pred.npy")
assert len(base) == len(val)
def wr(p): return weighted_rmse(y, p, cell[is_val], val["ADEP_mvt"], comp)

rtr = d[is_train & d["dq_no_aircraft_type"] & d["TAXITIME_SEC_mvt"].notna()].copy()
rva_m = val["dq_no_aircraft_type"].to_numpy()
rva = val[rva_m].copy()
print(f"regime rows: fit {len(rtr)}  val {rva_m.sum()}")
print(f"fit target   mean {rtr.TAXITIME_SEC_mvt.mean():,.0f} med {rtr.TAXITIME_SEC_mvt.median():,.0f} max {rtr.TAXITIME_SEC_mvt.max():,.0f}")
print(f"val target   mean {rva.TAXITIME_SEC_mvt.mean():,.0f} med {rva.TAXITIME_SEC_mvt.median():,.0f} max {rva.TAXITIME_SEC_mvt.max():,.0f}")

def plug(vals):
    p = base.copy(); p[rva_m] = np.maximum(np.asarray(vals, dtype="float64"), 0.0); return p

yr = rva["TAXITIME_SEC_mvt"].to_numpy()
def local(vals):
    return float(np.sqrt(np.mean((np.maximum(np.asarray(vals,dtype='float64'),0) - yr) ** 2)))

results = []
def show(name, vals):
    p = plug(vals); results.append((name, wr(p), local(vals)))
    print(f"  {name:<42} weighted {wr(p):7.1f}   regime-only RMSE {local(vals):9.0f}")

print("\nreference points")
show("model as-is (no branch)", base[rva_m])
show("constant = fit mean", np.full(rva_m.sum(), rtr.TAXITIME_SEC_mvt.mean()))
b = fit_regime_branch(d[is_train])
show("linear on proxy_taxi_sched (SHIPPED)", b["slope"]*rva["proxy_taxi_sched"].to_numpy()+b["intercept"])
show("oracle", yr)

# --- candidate 1: identity, y = proxy_sched (BLOCK==SCHED assumption) ---
print("\ncandidates")
ps_va = rva["proxy_taxi_sched"].to_numpy()
show("y = proxy_taxi_sched", np.nan_to_num(ps_va, nan=rtr.TAXITIME_SEC_mvt.mean()))

# --- candidate 2: robust/huber linear on proxy_sched ---
ok = rtr["proxy_taxi_sched"].notna()
X, Y = rtr.loc[ok, ["proxy_taxi_sched"]].to_numpy(), rtr.loc[ok, "TAXITIME_SEC_mvt"].to_numpy()
for name, mdl in [("huber", HuberRegressor(epsilon=1.35)), ("ridge", Ridge(alpha=1.0))]:
    mdl.fit(X, Y)
    v = mdl.predict(np.nan_to_num(rva[["proxy_taxi_sched"]].to_numpy(), nan=np.median(X)))
    show(f"{name} on proxy_taxi_sched", v)

# --- candidate 3: LightGBM on the regime rows, all available features ---
nums, cats = featurelist(d)
# these rows have NO flight-plan record at all, so every _flt-derived column is null
usable = [c for c in nums + cats if rtr[c].notna().mean() > 0.5]
dd = encode(d.copy(), [c for c in cats if c in usable])
rtr2, rva2 = dd[is_train & dd["dq_no_aircraft_type"] & dd["TAXITIME_SEC_mvt"].notna()], dd[is_val][rva_m]
print(f"  ({len(usable)} features populated on regime rows)")
for leaves, mcs, ne, lr in [(8, 20, 300, 0.05), (16, 10, 400, 0.05), (4, 30, 200, 0.08)]:
    m = lgb.LGBMRegressor(objective="regression", num_leaves=leaves, min_child_samples=mcs,
                          n_estimators=ne, learning_rate=lr, colsample_bytree=0.7,
                          subsample=0.8, subsample_freq=1, verbose=-1, random_state=0)
    m.fit(rtr2[usable], rtr2["TAXITIME_SEC_mvt"])
    show(f"lgbm leaves={leaves} mcs={mcs} n={ne}", m.predict(rva2[usable]))

# --- candidate 4: LightGBM on the residual delta = proxy_sched - y ---
for leaves, mcs, ne in [(8, 20, 300), (16, 10, 400)]:
    tgt = rtr2["proxy_taxi_sched"] - rtr2["TAXITIME_SEC_mvt"]
    ok2 = tgt.notna()
    m = lgb.LGBMRegressor(objective="regression", num_leaves=leaves, min_child_samples=mcs,
                          n_estimators=ne, learning_rate=0.05, colsample_bytree=0.7,
                          subsample=0.8, subsample_freq=1, verbose=-1, random_state=0)
    m.fit(rtr2.loc[ok2, usable], tgt[ok2])
    v = rva2["proxy_taxi_sched"].to_numpy() - m.predict(rva2[usable])
    v = np.where(np.isfinite(v), v, rtr.TAXITIME_SEC_mvt.mean())
    show(f"lgbm on delta, leaves={leaves} n={ne}", v)

print(f"\nbest: {min(results, key=lambda r: r[1])}")
