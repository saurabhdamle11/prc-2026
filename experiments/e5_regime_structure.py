#!/usr/bin/env python
"""E5: the missing-aircraft-type regime is bimodal -- can the two modes be told apart?

`y - proxy_taxi_sched` is exactly `SCHED_TIME - BLOCK_TIME`. Its quartiles on these rows are
(-4860, -5, +1): about half the rows have a block time stamped within seconds of the schedule
(so y == proxy_taxi_sched and the answer is known exactly), and the rest are ordinary taxis.
The single linear fit splits the difference between the two modes and is wrong for both.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness import load

d = load()
R = d[d["dq_no_aircraft_type"] & d["TAXITIME_SEC_mvt"].notna()].copy()
R["gap"] = R["TAXITIME_SEC_mvt"] - R["proxy_taxi_sched"]      # = SCHED - BLOCK
print(f"{len(R)} regime rows")
print("\n|SCHED - BLOCK| tolerance histogram")
for tol in [0, 1, 5, 30, 60, 300, 900]:
    m = R["gap"].abs() <= tol
    print(f"  <= {tol:>4}s : {m.sum():>5}  ({m.mean():6.1%})   mean y {R.loc[m,'TAXITIME_SEC_mvt'].mean():>8,.0f}")
print(f"  >  900s : {(R['gap'].abs()>900).sum():>5}  ({(R['gap'].abs()>900).mean():6.1%})"
      f"   mean y {R.loc[R['gap'].abs()>900,'TAXITIME_SEC_mvt'].mean():>8,.0f}")

R["mode"] = np.where(R["gap"].abs() <= 60, "block==sched", "ordinary")
print("\nthe two modes")
print(R.groupby("mode").apply(lambda x: pd.Series({
    "n": len(x), "mean_y": x.TAXITIME_SEC_mvt.mean(), "median_y": x.TAXITIME_SEC_mvt.median(),
    "sd_y": x.TAXITIME_SEC_mvt.std(), "median_proxy_sched": x.proxy_taxi_sched.median(),
}), include_groups=False).round(0).to_string())

print("\ncan the mode be predicted from observable fields?")
for c in ["ADEP_mvt", "RUNWAY_mvt", "stand_area", "FLIGHT_RULE_mvt"]:
    t = pd.crosstab(R[c], R["mode"], normalize="index")
    n = R[c].value_counts()
    t = t.assign(n=n).query("n >= 25").sort_values("block==sched", ascending=False)
    if len(t): print(f"\n  by {c} (>=25 rows):\n{t.round(3).head(10).to_string()}")

R["month"] = R["MVT_TIME_UTC_mvt"].dt.month
print("\n  by month:")
print(pd.crosstab(R["month"], R["mode"], normalize="index").round(3).assign(
      n=R["month"].value_counts()).to_string())
print("\n  by proxy_taxi_sched band:")
R["psb"] = pd.cut(R.proxy_taxi_sched, [-1e9, 1800, 3600, 5400, 7200, 10800, 1e9])
print(pd.crosstab(R["psb"], R["mode"], normalize="index").round(3).assign(
      n=R["psb"].value_counts()).to_string())
