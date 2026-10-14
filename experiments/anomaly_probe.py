#!/usr/bin/env python
"""What ARE the broken-label rows, and is the corruption visible in co-located ARR rows?"""
from __future__ import annotations
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import load_training, clean_movements, add_offblock_proxies, MAX_PLAUSIBLE_TAXI_SEC

raw = load_training()
print(f"all movements {len(raw):,}")
dep = add_offblock_proxies(clean_movements(raw, phase="DEP"))
arr = clean_movements(raw, phase="ARR")

noact = dep["dq_no_aircraft_type"]
impl  = dep["dq_implausible_target"]
inv   = impl & ~noact
print(f"\nDEP: no_actype {noact.sum()}  implausible {impl.sum()}  invisible(impl & has actype) {inv.sum()}")

print("\n--- the invisible population (broken label, aircraft type present) ---")
sub = dep[inv]
print(sub["ADEP_mvt"].value_counts().to_string())
print("target: ", sub["TAXITIME_SEC_mvt"].describe().round(0).to_dict())
print("target/86400 rounded:", pd.Series((sub['TAXITIME_SEC_mvt']/86400).round(1)).value_counts().head(8).to_dict())
print("has AOBT_3:", sub["AOBT_3_flt"].notna().mean().round(3), " has flight record:", (~sub["dq_no_flight_record"]).mean().round(3))
print("proxy_taxi_aobt3 on these:", sub["proxy_taxi_aobt3"].describe().round(0).to_dict())

print("\n--- the no_actype population ---")
sub2 = dep[noact]
print(sub2["ADEP_mvt"].value_counts().to_string())
print("target:", sub2["TAXITIME_SEC_mvt"].describe().round(0).to_dict())
print("has flight record:", (~sub2["dq_no_flight_record"]).mean().round(4))
print("has AOBT_3:", sub2["AOBT_3_flt"].notna().mean().round(4), " has SCHED:", sub2["SCHED_TIME_UTC_mvt"].notna().mean().round(4))
print("has RUNWAY:", sub2["RUNWAY_mvt"].notna().mean().round(3), " has STAND:", sub2["STAND_mvt"].notna().mean().round(3))
print("frac of no_actype that are implausible:", (sub2['TAXITIME_SEC_mvt']>MAX_PLAUSIBLE_TAXI_SEC).mean().round(3))

# --- do ARR rows show the same corruption at the same airport/time? ---
print("\n--- ARR taxi-in distribution (this is LABELLED in ranking.parquet) ---")
arr_t = arr["TAXITIME_SEC_mvt"]
print(arr_t.describe().round(0).to_dict())
print("ARR over 2h:", (arr_t > MAX_PLAUSIBLE_TAXI_SEC).sum(), f"({(arr_t>MAX_PLAUSIBLE_TAXI_SEC).mean():.5%})")
print("ARR no actype:", arr['AIRCRAFT_TYPE_mvt'].isna().sum())

# co-occurrence: for each broken DEP row, how many broken ARR rows in +/-6h at same airport?
def near_broken_arr(dep_sub, hours=6):
    out = []
    ab = arr[arr_t > MAX_PLAUSIBLE_TAXI_SEC]
    for apt, g in dep_sub.groupby("ADEP_mvt", observed=True):
        cand = ab[ab["ADES_mvt"] == apt]["MVT_TIME_UTC_mvt"]
        if cand.empty:
            out.append(pd.Series(0, index=g.index)); continue
        c = np.sort(cand.dt.tz_convert('UTC').dt.tz_localize(None).to_numpy('datetime64[ns]').astype('int64'))
        q = g["MVT_TIME_UTC_mvt"].dt.tz_convert('UTC').dt.tz_localize(None).to_numpy('datetime64[ns]').astype('int64')
        half = hours*3600*10**9
        out.append(pd.Series(np.searchsorted(c, q+half,'right') - np.searchsorted(c, q-half,'left'), index=g.index))
    return pd.concat(out).reindex(dep_sub.index).fillna(0)

for name, sel in [("invisible", inv), ("no_actype", noact), ("normal sample", ~(noact|impl))]:
    s = dep[sel]
    if name == "normal sample":
        s = s.sample(20000, random_state=0)
    n = near_broken_arr(s)
    print(f"  {name:<15} n={len(s):>6,}  mean broken-ARR within +/-6h at same airport: {n.mean():.2f}   share with >=1: {(n>0).mean():.1%}")
