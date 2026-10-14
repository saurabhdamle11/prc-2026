#!/usr/bin/env python
"""E18: are the three feature families the catalogue is missing actually available and useful?

Checked with the project's controlled-comparison rule: a raw correlation with the target is
confounded by airport, stand and runway, so every candidate is measured against the *residual*
after removing the (airport, stand_area, runway) median.
"""
import sys, glob; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cols = ["MVT_ID_mvt","PHASE_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","TAXITIME_SEC_mvt",
        "STAND_mvt","RUNWAY_mvt","AOBT_3_flt","AIRCRAFT_OPERATOR_flt","SCHED_TIME_UTC_mvt"]
tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("Dataset/training_*.parquet"))],
               ignore_index=True)
rk = pd.read_parquet("Dataset/ranking.parquet", columns=cols)

print("=== availability of the inputs these would need ===")
for nm, d in [("train", tr), ("ranking", rk)]:
    a = d[d.PHASE_mvt == "ARR"]
    print(f"{nm:<8} ARR rows {len(a):,}  STAND present {a.STAND_mvt.notna().mean():.1%}  "
          f"TAXITIME present {a.TAXITIME_SEC_mvt.notna().mean():.1%}  (in-block = MVT_TIME + TAXITIME)")

# ---------- 1. rotation: link each departure to the inbound that vacated its stand -------
def rotation(d):
    dep = d[d.PHASE_mvt == "DEP"].copy()
    arr = d[(d.PHASE_mvt == "ARR") & d.STAND_mvt.notna() & d.TAXITIME_SEC_mvt.notna()].copy()
    arr["inblock"] = arr.MVT_TIME_UTC_mvt + pd.to_timedelta(arr.TAXITIME_SEC_mvt, unit="s")
    arr["apt"] = arr.ADES_mvt
    dep["anchor"] = dep.AOBT_3_flt.fillna(dep.SCHED_TIME_UTC_mvt)
    out = np.full(len(dep), np.nan)
    for key, g in dep.groupby([dep.ADEP_mvt, dep.STAND_mvt], observed=True):
        cand = arr[(arr.apt == key[0]) & (arr.STAND_mvt == key[1])]
        if cand.empty: continue
        t = np.sort(cand.inblock.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]"))
        q = g.anchor.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
        i = np.searchsorted(t, q, "left") - 1                     # most recent inbound before push
        ok = i >= 0
        pos = dep.index.get_indexer(g.index)
        v = np.full(len(g), np.nan)
        v[ok] = (q[ok] - t[i[ok]]) / np.timedelta64(1, "s")
        out[pos] = v
    dep["stand_occupied_sec"] = out
    return dep

dep = rotation(tr)
m = dep.stand_occupied_sec.notna() & dep.stand_occupied_sec.between(0, 86400)
print(f"\n=== 1. rotation (inbound that last vacated this stand) ===")
print(f"matched on {m.mean():.1%} of training departures; turnaround quantiles "
      f"{dep.loc[m,'stand_occupied_sec'].quantile([.1,.5,.9]).round(0).to_dict()}")
s = dep[m & dep.TAXITIME_SEC_mvt.notna()].copy()
s["grp"] = s.ADEP_mvt.astype(str)+"|"+s.STAND_mvt.astype(str)+"|"+s.RUNWAY_mvt.astype(str)
s["resid"] = s.TAXITIME_SEC_mvt - s.groupby("grp")["TAXITIME_SEC_mvt"].transform("median")
print(f"  corr(turnaround, taxi)              {s.stand_occupied_sec.corr(s.TAXITIME_SEC_mvt):+.4f}")
print(f"  corr(turnaround, residual)          {s.stand_occupied_sec.corr(s.resid):+.4f}   <- the controlled one")
print("  residual by turnaround band:")
s["b"] = pd.cut(s.stand_occupied_sec, [0,1800,3600,7200,21600,86400])
print(s.groupby("b", observed=True)["resid"].agg(n="size", mean="mean").round(1).to_string())

# ---------- 2. runway configuration in use ----------------------------------------------
print("\n=== 2. runway configuration (set of runways in departure use) ===")
d = tr[(tr.PHASE_mvt=="DEP") & tr.RUNWAY_mvt.notna()].copy()
d["hr"] = d.MVT_TIME_UTC_mvt.dt.floor("h")
cfg = (d.groupby(["ADEP_mvt","hr"])["RUNWAY_mvt"].apply(lambda x: "+".join(sorted(set(x)))).rename("cfg"))
d = d.join(cfg, on=["ADEP_mvt","hr"])
print(f"  distinct (airport, config) combinations: {d.groupby('ADEP_mvt')['cfg'].nunique().to_dict()}")
d["grp"] = d.ADEP_mvt.astype(str)+"|"+d.STAND_mvt.astype(str)+"|"+d.RUNWAY_mvt.astype(str)
d["resid"] = d.TAXITIME_SEC_mvt - d.groupby("grp")["TAXITIME_SEC_mvt"].transform("median")
v = d.groupby(["ADEP_mvt","cfg"])["resid"].agg(n="size", mean="mean")
v = v[v.n >= 500]
print(f"  residual spread across configs, per airport (rows>=500):")
print(v.groupby(level=0)["mean"].agg(["min","max","count"]).round(1).to_string())

# ---------- 3. how much (stand, runway) structure is left on the table -------------------
print("\n=== 3. static (stand_area, runway) taxi level ===")
t = pd.read_parquet("data/processed/train_features.parquet",
                    columns=["TAXITIME_SEC_mvt","stand_area","RUNWAY_mvt","ADEP_mvt","dq_no_aircraft_type"])
t = t[t.TAXITIME_SEC_mvt.notna() & ~t.dq_no_aircraft_type]
for keys in [["ADEP_mvt"], ["stand_area"], ["stand_area","RUNWAY_mvt"]]:
    g = t.groupby(keys, observed=True)["TAXITIME_SEC_mvt"]
    pred = g.transform("median")
    print(f"  median by {str(keys):<34} n_groups {g.ngroups:>5}  residual sd {np.sqrt(((t.TAXITIME_SEC_mvt-pred)**2).mean()):7.1f}s")
