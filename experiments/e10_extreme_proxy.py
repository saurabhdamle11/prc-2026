#!/usr/bin/env python
"""E10: when the ADS-B proxy is implausibly large, is it right?

A date error in a movement record plausibly shifts BOTH timestamps it derives from, so a row
whose `MVT_TIME - AOBT_3` is 24 h may simply be telling the truth about an equally shifted
BLOCK_TIME. If so, the model should trust the proxy there instead of regressing to a normal
taxi -- which is precisely what a tree cannot do, since it can never predict outside the range
of targets it was fitted on.

Sized on the actual scored set, not just on training: the rule is only worth shipping if
ranking.parquet contains rows it fires on.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

d = pd.read_parquet("data/processed/train_features.parquet",
    columns=["TAXITIME_SEC_mvt","proxy_taxi_aobt3","proxy_taxi_sched","ADEP_mvt",
             "dq_no_aircraft_type","MVT_TIME_UTC_mvt"])
d = d[d.TAXITIME_SEC_mvt.notna()]
print("TRAINING: rows by proxy_taxi_aobt3 band\n")
bands = [(1800,3600),(3600,7200),(7200,14400),(14400,43200),(43200,10**9)]
print(f"{'band':<22}{'n':>8}{'E[y]':>10}{'median y':>10}{'corr(y,proxy)':>15}{'median y-proxy':>16}{'|y-proxy|<300':>15}")
for lo,hi in bands:
    m = d.proxy_taxi_aobt3.between(lo,hi,inclusive="left")
    s = d[m]
    if not len(s): continue
    print(f"[{lo:>6},{hi:>9})   {len(s):>8,}{s.TAXITIME_SEC_mvt.mean():>10,.0f}"
          f"{s.TAXITIME_SEC_mvt.median():>10,.0f}{s.TAXITIME_SEC_mvt.corr(s.proxy_taxi_aobt3):>15.3f}"
          f"{(s.TAXITIME_SEC_mvt-s.proxy_taxi_aobt3).median():>16,.0f}"
          f"{((s.TAXITIME_SEC_mvt-s.proxy_taxi_aobt3).abs()<300).mean():>15.1%}")

print("\nrows with proxy > 7200 (the extrapolation zone):")
hi = d[d.proxy_taxi_aobt3 > 7200]
print(f"  n={len(hi)}  ({len(hi)/len(d):.5%} of train)   no_actype: {hi.dq_no_aircraft_type.sum()}")
print(f"  |y - proxy| < 300s on {((hi.TAXITIME_SEC_mvt-hi.proxy_taxi_aobt3).abs()<300).mean():.1%}")
print(f"  y < 3600 (proxy is simply wrong) on {(hi.TAXITIME_SEC_mvt<3600).mean():.1%}")
print(f"  RMSE if we predict the proxy : {np.sqrt(np.mean((hi.proxy_taxi_aobt3-hi.TAXITIME_SEC_mvt)**2)):,.0f}")
print(f"  RMSE if we predict E[y]      : {np.sqrt(np.mean((hi.TAXITIME_SEC_mvt.mean()-hi.TAXITIME_SEC_mvt)**2)):,.0f}")
print(f"  RMSE if we predict a normal ~1000s: {np.sqrt(np.mean((1000-hi.TAXITIME_SEC_mvt)**2)):,.0f}")
for t in [0.25,0.5,0.75,1.0]:
    p = t*hi.proxy_taxi_aobt3 + (1-t)*1000
    print(f"    blend {t:.2f}*proxy + {1-t:.2f}*1000 : RMSE {np.sqrt(np.mean((p-hi.TAXITIME_SEC_mvt)**2)):,.0f}")

r = pd.read_parquet("data/processed/ranking_features.parquet",
    columns=["proxy_taxi_aobt3","ADEP_mvt","dq_no_aircraft_type","MVT_TIME_UTC_mvt"])
print(f"\nRANKING (the scored set): {len(r):,} rows")
for th in [3600, 7200, 14400, 43200]:
    m = r.proxy_taxi_aobt3 > th
    print(f"  proxy > {th:>6}s : {m.sum():>5} rows ({m.mean():.5%})   "
          f"of which no_actype {int(r.loc[m,'dq_no_aircraft_type'].sum())}")
m = r.proxy_taxi_aobt3 > 7200
print(f"\n  those rows by airport: {r.loc[m,'ADEP_mvt'].value_counts().to_dict()}")
print(f"  by month: {r.loc[m,'MVT_TIME_UTC_mvt'].dt.month.value_counts().to_dict()}")
