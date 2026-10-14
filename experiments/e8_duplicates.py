#!/usr/bin/env python
"""E8: do the unmatched (no aircraft type) records have a matched twin we can join to?

These rows carry no NM flight record at all, so every _flt field is null and the model's
dominant feature is missing. If the airport wrote a second, properly-matched movement for the
same physical flight, joining on flight number + time would recover AOBT_3 and with it a
normal prediction. Checked against both same-airport DEP twins and the destination's ARR row.
"""
import sys, glob; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cols = ["MVT_ID_mvt","FLIGHT_mvt","PHASE_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt","AIRCRAFT_TYPE_mvt","TAXITIME_SEC_mvt","STAND_mvt","RUNWAY_mvt"]
tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("Dataset/training_*.parquet"))],
               ignore_index=True)
dep = tr[tr.PHASE_mvt == "DEP"]
noact = dep[dep.AIRCRAFT_TYPE_mvt.isna()]
print(f"{len(noact)} unmatched DEP rows; flight number present on {noact.FLIGHT_mvt.notna().mean():.1%}")
print("sample flight numbers:", noact.FLIGHT_mvt.dropna().head(8).tolist())

# twin = another DEP row, same airport, same flight number, takeoff within 6 h
others = dep[dep.AIRCRAFT_TYPE_mvt.notna()]
key = ["ADEP_mvt", "FLIGHT_mvt"]
m = noact.merge(others, on=key, suffixes=("", "_t"))
m["dt"] = (m.MVT_TIME_UTC_mvt_t - m.MVT_TIME_UTC_mvt).dt.total_seconds().abs()
near = m[m.dt <= 6*3600]
print(f"\nsame-airport DEP twins within 6h: {near.MVT_ID_mvt.nunique()} of {len(noact)} rows "
      f"({near.MVT_ID_mvt.nunique()/len(noact):.1%})")
if len(near):
    best = near.sort_values("dt").drop_duplicates("MVT_ID_mvt")
    print(f"  median dt {best.dt.median():.0f}s; block times identical on "
          f"{(best.BLOCK_TIME_UTC_mvt == best.BLOCK_TIME_UTC_mvt_t).mean():.1%}")
    print(f"  twin taxi vs own taxi: corr {best.TAXITIME_SEC_mvt.corr(best.TAXITIME_SEC_mvt_t):.3f}")

# does the unmatched row duplicate a row at the DESTINATION as an ARR?
arr = tr[tr.PHASE_mvt == "ARR"]
m2 = noact.merge(arr, left_on=["ADES_mvt","FLIGHT_mvt"], right_on=["ADES_mvt","FLIGHT_mvt"],
                 suffixes=("", "_a"))
m2["dt"] = (m2.MVT_TIME_UTC_mvt_a - m2.MVT_TIME_UTC_mvt).dt.total_seconds()
ok = m2[(m2.dt > 0) & (m2.dt < 12*3600)]
print(f"\nmatching ARR at destination within 12h: {ok.MVT_ID_mvt.nunique()} rows "
      f"({ok.MVT_ID_mvt.nunique()/len(noact):.1%})")

# how often is the flight number itself reused / meaningful
print(f"\nunmatched rows sharing a flight number with ANY other movement: "
      f"{noact.FLIGHT_mvt.isin(tr.loc[tr.AIRCRAFT_TYPE_mvt.notna(),'FLIGHT_mvt']).mean():.1%}")
