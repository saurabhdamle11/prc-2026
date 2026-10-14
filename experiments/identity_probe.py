#!/usr/bin/env python
"""Does BLOCK_TIME_UTC_mvt coincide exactly with any field that survives into ranking.parquet?

If it does, the target is not a regression problem on those rows -- it is an identity, and the
task reduces to classifying which rows the identity holds on.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import load_training, clean_movements

raw = load_training()
dep = clean_movements(raw, phase="DEP")
b = dep["BLOCK_TIME_UTC_mvt"]
print(f"DEP rows {len(dep):,}   BLOCK_TIME present {b.notna().mean():.4%}\n")

cands = ["SCHED_TIME_UTC_mvt","AOBT_3_flt","EOBT_1_flt","IOBT_flt","LOBT_flt"]
print(f"{'field':<22}{'present':>10}{'exact ==':>12}{'share':>10}{'within 60s':>12}{'share':>10}")
for c in cands:
    if c not in dep.columns: continue
    both = b.notna() & dep[c].notna()
    diff = (b - dep[c]).dt.total_seconds()
    ex = (both & (diff == 0)).sum()
    w60 = (both & (diff.abs() <= 60)).sum()
    print(f"{c:<22}{both.sum():>10,}{ex:>12,}{ex/len(dep):>10.3%}{w60:>12,}{w60/len(dep):>10.3%}")

print("\n--- exact BLOCK==SCHED, broken out ---")
eq = (b - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds() == 0
print(f"overall {eq.mean():.3%}  n={eq.sum():,}")
print("\nby airport:")
print((dep.groupby("ADEP_mvt", observed=True).apply(
    lambda g: pd.Series({"n": len(g), "eq_rate": ((g['BLOCK_TIME_UTC_mvt']-g['SCHED_TIME_UTC_mvt']).dt.total_seconds()==0).mean()}),
    include_groups=False).sort_values("eq_rate", ascending=False)).to_string())
print("\nby no_actype flag:")
print(dep.groupby("dq_no_aircraft_type")[[]].size().to_string())
for flag in [True, False]:
    m = dep["dq_no_aircraft_type"] == flag
    print(f"  no_actype={flag}: eq rate {eq[m].mean():.3%} of {m.sum():,}")
print("\nno flight record?")
for flag in [True, False]:
    m = dep["dq_no_flight_record"] == flag
    print(f"  no_flight_record={flag}: eq rate {eq[m].mean():.3%} of {m.sum():,}")

print("\ntarget distribution where BLOCK==SCHED vs not:")
print(dep.loc[eq,"TAXITIME_SEC_mvt"].describe().round(0).to_dict())
print(dep.loc[~eq,"TAXITIME_SEC_mvt"].describe().round(0).to_dict())
