#!/usr/bin/env python
"""E21: `jul LIRF` is 35.6% of the weighted MSE. What is actually in it?

E19 re-measured the budget under the current model and one cell dominates everything else by
a factor of five (35.6% against the next cell's 7.4%, RMSE 779.5s against a 150-300s norm).
Before modelling anything, establish *which rows* in that cell carry it, and whether the cell
is hard in general or July-2025 specifically -- the protocol scores July 2025 but the
leaderboard scores July 2026, so a one-off 2025 disruption would make this cell a phantom.

Reuses e19's cached predictions: no refit.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.validation import mirrored_split

d = pd.read_parquet("data/processed/train_features.parquet")
is_train, is_val, cell = mirrored_split(d)
val = d[is_val].copy()
pred = np.load("data/processed/exp_cache/r4_pred.npy")
assert len(pred) == len(val)
val["pred"] = pred
val["cell"] = cell[is_val].to_numpy()
val["resid"] = val["pred"] - val["TAXITIME_SEC_mvt"]
val["sq"] = val["resid"] ** 2

L = val[(val.cell == "jul") & (val.ADEP_mvt == "LIRF")].copy()
print(f"=== jul LIRF: {len(L):,} rows, RMSE {np.sqrt(L.sq.mean()):.1f}s ===\n")

noact = L.dq_no_aircraft_type.to_numpy()
impl = L.dq_implausible_target.to_numpy()
pops = {"normal": ~(noact | impl), "missing aircraft type": noact, "invisible": impl & ~noact}
tot = L.sq.sum()
print(f"{'population':<24} {'n':>7} {'rmse':>9} {'share of cell MSE':>19}")
for nm, m in pops.items():
    print(f"{nm:<24} {int(m.sum()):>7,} {np.sqrt(L.sq[m].mean()):>9.1f} {L.sq[m].sum()/tot:>18.1%}")

N = L[pops["normal"]]
print(f"\n--- inside normal jul LIRF ({len(N):,} rows, RMSE {np.sqrt(N.sq.mean()):.1f}s) ---")
bands = [(0, 600), (600, 1200), (1200, 1800), (1800, 2700), (2700, 3600), (3600, 1e9)]
ntot = N.sq.sum()
print(f"{'true taxi band':<22} {'n':>7} {'share rows':>11} {'rmse':>9} {'mean resid':>11} {'share MSE':>10}")
for lo, hi in bands:
    m = (N.TAXITIME_SEC_mvt >= lo) & (N.TAXITIME_SEC_mvt < hi)
    if m.sum() == 0: continue
    print(f"[{lo:>5}, {hi:>7.0f})      {int(m.sum()):>7,} {m.mean():>10.1%} "
          f"{np.sqrt(N.sq[m].mean()):>9.1f} {N.resid[m].mean():>11.1f} {N.sq[m].sum()/ntot:>9.1%}")

print("\n--- is LIRF hard in general, or July-2025 specifically? ---")
lirf = d[(d.ADEP_mvt == "LIRF") & d.TAXITIME_SEC_mvt.notna()].copy()
lirf["month"] = lirf.MVT_TIME_UTC_mvt.dt.month
tm = trainable_mask(lirf)
g = lirf[tm].groupby("month")["TAXITIME_SEC_mvt"]
q = lirf[tm].groupby("month")["TAXITIME_SEC_mvt"].quantile([0.5, 0.9, 0.99]).unstack()
tbl = pd.DataFrame({"n": g.size(), "mean": g.mean(), "p50": q[0.5], "p90": q[0.9],
                    "p99": q[0.99], "frac>1800s": lirf[tm].assign(
                        f=lambda x: x.TAXITIME_SEC_mvt > 1800).groupby("month")["f"].mean()})
print(tbl.round(1).to_string())

print("\n--- the same, for every airport: fraction of taxis over 1800s by month ---")
a = d[d.TAXITIME_SEC_mvt.notna()].copy()
a = a[trainable_mask(a)]
a["month"] = a.MVT_TIME_UTC_mvt.dt.month
piv = (a.assign(f=lambda x: x.TAXITIME_SEC_mvt > 1800)
        .pivot_table(index="ADEP_mvt", columns="month", values="f", aggfunc="mean"))
print((piv * 100).round(1).to_string())

print("\n--- LIRF runway/stand mix, July vs rest of year (trainable rows) ---")
lt = lirf[tm].copy()
lt["is_jul"] = lt.month == 7
for col in ["RUNWAY_mvt", "stand_area"]:
    top = lt[col].value_counts().head(8).index
    sub = lt[lt[col].isin(top)]
    p = sub.pivot_table(index=col, columns="is_jul", values="TAXITIME_SEC_mvt",
                        aggfunc=["size", "mean"])
    print(f"\n{col}:")
    print(p.round(0).to_string())
