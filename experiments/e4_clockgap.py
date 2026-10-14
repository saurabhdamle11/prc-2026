#!/usr/bin/env python
"""E4: can the ADS-B/airport timing offset be observed from the arrival side?

The departure residual the model is really fitting is `AOBT_3_flt - BLOCK_TIME_UTC_mvt` -- how
long after the airport logged off-block the flown trajectory first shows movement. That is
unobservable at prediction time (BLOCK_TIME is blank in ranking.parquet).

The arrival rows offer a possible substitute: `ARVT_3_flt - MVT_TIME_UTC_mvt` on ARR rows is
the same two clocks compared (flown trajectory vs airport movement log), and BOTH are present
in ranking.parquet. If the two offsets share a time-varying component, a rolling arrival-side
gap is a legitimate proxy for the departure-side one.
"""
import sys, glob; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cols = ["PHASE_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt",
        "AOBT_3_flt","ARVT_3_flt","TAXITIME_SEC_mvt"]
tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("Dataset/training_*.parquet"))],
               ignore_index=True)
rk = pd.read_parquet("Dataset/ranking.parquet", columns=cols)

for nm, df in [("train", tr), ("ranking", rk)]:
    a = df[df.PHASE_mvt=="ARR"]
    print(f"{nm:<8} ARR rows {len(a):,}  ARVT_3 present {a.ARVT_3_flt.notna().mean():.3%}  "
          f"MVT_TIME present {a.MVT_TIME_UTC_mvt.notna().mean():.3%}")

dep = tr[tr.PHASE_mvt=="DEP"].copy()
dep["off"] = (dep.AOBT_3_flt - dep.BLOCK_TIME_UTC_mvt).dt.total_seconds()
arr = tr[tr.PHASE_mvt=="ARR"].copy()
arr["gap"] = (arr.ARVT_3_flt - arr.MVT_TIME_UTC_mvt).dt.total_seconds()
print(f"\ndeparture offset AOBT_3-BLOCK : {dep.off.describe(percentiles=[.1,.5,.9]).round(0).to_dict()}")
print(f"arrival gap ARVT_3-MVT_TIME   : {arr.gap.describe(percentiles=[.1,.5,.9]).round(0).to_dict()}")

dep["d"] = dep.MVT_TIME_UTC_mvt.dt.floor("D"); dep["apt"] = dep.ADEP_mvt
arr["d"] = arr.MVT_TIME_UTC_mvt.dt.floor("D"); arr["apt"] = arr.ADES_mvt
cl = lambda s: s[(s > s.quantile(.01)) & (s < s.quantile(.99))]
D = dep.groupby(["apt","d"])["off"].median().rename("dep_off")
A = arr.groupby(["apt","d"])["gap"].median().rename("arr_gap")
j = pd.concat([D, A], axis=1).dropna()
print(f"\nper (airport, day) medians, n={len(j):,}")
print(f"  pooled corr(dep_off, arr_gap) = {j.dep_off.corr(j.arr_gap):.3f}")
w = j.reset_index().groupby("apt").apply(
    lambda g: pd.Series({"n": len(g), "corr": g.dep_off.corr(g.arr_gap),
                         "dep_off_sd": g.dep_off.std(), "arr_gap_sd": g.arr_gap.std()}),
    include_groups=False)
print(w.round(3).to_string())
print("\nwithin-airport (airport mean removed):")
k = j.reset_index()
for c in ["dep_off","arr_gap"]:
    k[c] = k[c] - k.groupby("apt")[c].transform("mean")
print(f"  corr = {k.dep_off.corr(k.arr_gap):.3f}")
