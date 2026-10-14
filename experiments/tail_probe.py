#!/usr/bin/env python
"""Is the 7200s exclusion cut throwing away learnable signal?"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

d = pd.read_parquet("data/processed/train_features.parquet",
    columns=["TAXITIME_SEC_mvt","proxy_taxi_sched","proxy_taxi_aobt3","proxy_taxi_eobt1",
             "dq_no_aircraft_type","dq_implausible_target","ADEP_mvt","MVT_TIME_UTC_mvt",
             "takeoff_vs_sched_sec","offblock_delay_sec"])
y = d["TAXITIME_SEC_mvt"]
has_act = ~d["dq_no_aircraft_type"]

print("target distribution, rows WITH an aircraft type")
for lo, hi in [(0,1800),(1800,3600),(3600,5400),(5400,7200),(7200,10800),(10800,21600),(21600,10**9)]:
    m = has_act & y.between(lo, hi, inclusive="left")
    print(f"  [{lo:>6},{hi:>9}) n={m.sum():>9,}  {m.mean():8.5%}")

print("\ncorrelation of target with the schedule proxy, by target band (rows with aircraft type)")
for lo, hi, lbl in [(0,3600,"normal <1h"),(3600,7200,"1-2h (trainable)"),(7200,10**9,"over 2h (EXCLUDED)")]:
    m = has_act & y.between(lo,hi,inclusive="left")
    s = d[m]
    ok = s["proxy_taxi_sched"].notna()
    print(f"  {lbl:<20} n={m.sum():>8,}  corr(y, proxy_sched)={s.loc[ok,'proxy_taxi_sched'].corr(s.loc[ok,'TAXITIME_SEC_mvt']):.3f}"
          f"   median proxy_sched {s['proxy_taxi_sched'].median():>8.0f}  median y {s['TAXITIME_SEC_mvt'].median():>8.0f}"
          f"   median proxy_aobt3 {s['proxy_taxi_aobt3'].median():>8.0f}")

print("\nno_actype rows: how well does each proxy explain the target?")
s = d[d["dq_no_aircraft_type"]]
for c in ["proxy_taxi_sched","takeoff_vs_sched_sec"]:
    ok = s[c].notna()
    print(f"  corr(y,{c}) = {s.loc[ok,c].corr(s.loc[ok,'TAXITIME_SEC_mvt']):.3f}  (n={ok.sum()})")
print("  spearman proxy_sched:", round(s['proxy_taxi_sched'].corr(s['TAXITIME_SEC_mvt'], method='spearman'),3))
print("  y quantiles:", s['TAXITIME_SEC_mvt'].quantile([.1,.25,.5,.75,.9,.99]).round(0).to_dict())
print("  proxy_sched quantiles:", s['proxy_taxi_sched'].quantile([.1,.25,.5,.75,.9,.99]).round(0).to_dict())
print("  residual y - proxy_sched:", (s['TAXITIME_SEC_mvt']-s['proxy_taxi_sched']).describe().round(0).to_dict())
