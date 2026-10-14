#!/usr/bin/env python
"""E34: does an ADS-B pushback detector actually beat AOBT_3_flt? The kill-switch.

The whole ADS-B project rests on one claim: `AOBT_3_flt` marks the first *sustained* movement
and therefore misses an apron hold entirely (+2,758 s median on taxis over an hour, `e22`),
while the raw surface track shows the **pushback itself**. This tests that on real data.

Training rows carry `BLOCK_TIME_UTC_mvt`, so both estimators can be scored against the truth
on the same rows. The bar is `AOBT_3_flt`: **422.9 s** weighted RMSE overall (`e31`).

Detector: for each departure, take that callsign's surface points at that airport in the three
hours before takeoff. The aircraft sits at its stand (ground speed ~0), pushes back, possibly
holds, then taxis. Off-block is the **end of the last long stationary period** before takeoff —
i.e. the moment it first moved off the stand, which is what the target measures and what a
sustained-movement rule discards.
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DAY = "2025-01-15"
STOP_KT = 1.5        # at or below this the aircraft is not moving
MOVE_KT = 3.0        # above this it is definitely moving
MIN_STOP_S = 300     # a stand occupancy is at least this long
LOOKBACK_S = 3 * 3600

g = pd.read_parquet(f"data/external/adsb_ground/{DAY}.parquet")
g = g[g.callsign.notna()].copy()
g["gs"] = g.gs.fillna(0.0)

cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "TAXITIME_SEC_mvt", "CALLSIGN_flt", "AOBT_3_flt", "AIRCRAFT_TYPE_mvt"]
tr = pd.read_parquet("Dataset/training_2025-01-01_2025-02-01.parquet", columns=cols)
d0 = pd.Timestamp(DAY, tz="UTC")
dep = tr[(tr.PHASE_mvt == "DEP")
         & (tr.MVT_TIME_UTC_mvt >= d0) & (tr.MVT_TIME_UTC_mvt < d0 + pd.Timedelta(days=1))
         & tr.CALLSIGN_flt.notna() & tr.BLOCK_TIME_UTC_mvt.notna()].copy()
# NB: pandas 3.0 keeps second-resolution datetimes from parquet, so `.astype("int64")`
# yields SECONDS, not nanoseconds. Subtracting the epoch is resolution-independent.
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
dep["mvt_s"] = (dep.MVT_TIME_UTC_mvt - EPOCH).dt.total_seconds()
dep["block_s"] = (dep.BLOCK_TIME_UTC_mvt - EPOCH).dt.total_seconds()

COVERED = sorted(g.airport.unique())
print(f"{DAY}: {len(dep):,} DEP rows with a callsign and a known BLOCK_TIME")
print(f"adsb.lol surface coverage at: {', '.join(COVERED)}\n")

# --- coverage: does this departure have surface points near its takeoff? ---------------
gg = {k: v.sort_values("time") for k, v in g.groupby(["airport", "callsign"], observed=True)}
rows = []
for r in dep.itertuples():
    sub = gg.get((r.ADEP_mvt, r.CALLSIGN_flt))
    if sub is None:
        rows.append((r.MVT_ID_mvt, np.nan, 0, False, np.nan)); continue
    t = sub.time.to_numpy(); v = sub.gs.to_numpy()
    m = (t >= r.mvt_s - LOOKBACK_S) & (t <= r.mvt_s + 120)
    if m.sum() < 3:
        rows.append((r.MVT_ID_mvt, np.nan, int(m.sum()), False, np.nan)); continue
    t, v = t[m], v[m]
    # walk back from takeoff: find the last stationary run of >= MIN_STOP_S
    moving = v > MOVE_KT
    stopped = v <= STOP_KT
    off = np.nan
    i = len(t) - 1
    while i > 0:
        if stopped[i]:
            j = i
            while j > 0 and stopped[j - 1]:
                j -= 1
            if t[i] - t[j] >= MIN_STOP_S:
                # off-block = first sample after this stand occupancy ends
                off = t[i + 1] if i + 1 < len(t) else t[i]
                break
            i = j - 1
        else:
            i -= 1
    if not np.isfinite(off):
        mv = np.flatnonzero(moving)
        off = t[mv[0]] if len(mv) else np.nan
    # can the off-block even be seen? only if the track begins with the aircraft stopped,
    # well before takeoff. If the transponder came on mid-pushback the moment is unobservable.
    starts_stopped = bool(stopped[:3].any()) and (t[0] <= r.mvt_s - 300)
    rows.append((r.MVT_ID_mvt, off, int(m.sum()), starts_stopped, float(t[0] - r.mvt_s)))

det = pd.DataFrame(rows, columns=["MVT_ID_mvt", "offblock_adsb", "n_pts",
                                  "starts_stopped", "first_pt_rel_takeoff"])
dep = dep.merge(det, on="MVT_ID_mvt", how="left")
dep_c = dep[dep.ADEP_mvt.isin(COVERED)]
print("=== coverage on covered airports ===")
cov = dep_c.groupby("ADEP_mvt").agg(dep_rows=("MVT_ID_mvt", "size"),
                                    with_track=("offblock_adsb", lambda s: s.notna().sum()))
cov["coverage"] = cov.with_track / cov.dep_rows
print(cov.assign(coverage=(cov.coverage * 100).round(1)).to_string())
print(f"\noverall on covered airports: {dep_c.offblock_adsb.notna().mean():.1%}")

# --- accuracy against the truth ---------------------------------------------------------
print("\n=== is the off-block moment even observable? ===")
h = dep_c[dep_c.offblock_adsb.notna()]
print(f"  departures with any track:        {len(h):,}")
print(f"  track begins stationary (usable): {int(h.starts_stopped.sum()):,} "
      f"({h.starts_stopped.mean():.1%} of tracked)")
print(f"  first point vs takeoff (median):  {h.first_pt_rel_takeoff.median():,.0f}s")

ok = dep_c[dep_c.offblock_adsb.notna()].copy()
ok["err_adsb"] = ok.offblock_adsb - ok.block_s
ok["err_aobt3"] = (ok.AOBT_3_flt - EPOCH).dt.total_seconds() - ok.block_s
ok.loc[ok.AOBT_3_flt.isna(), "err_aobt3"] = np.nan
print(f"\n=== error against true BLOCK_TIME, on {len(ok):,} matched departures ===")
print(f"{'estimator':<22} {'n':>7} {'RMSE':>9} {'median':>9} {'p90|err|':>9}")
for name, c in [("ADS-B detector", "err_adsb"), ("AOBT_3_flt", "err_aobt3")]:
    e = ok[c].dropna()
    print(f"{name:<22} {len(e):>7,} {np.sqrt((e**2).mean()):>9.1f} {e.median():>9.1f} "
          f"{e.abs().quantile(.9):>9.1f}")

print("\n=== by true taxi band — does the detector stay flat where AOBT_3 blows up? ===")
print(f"{'true taxi':<20} {'n':>6} {'ADS-B rmse':>12} {'AOBT_3 rmse':>12} {'ADS-B med':>10} {'AOBT_3 med':>11}")
for lo, hi in [(0, 1200), (1200, 1800), (1800, 2700), (2700, 1e9)]:
    m = (ok.TAXITIME_SEC_mvt >= lo) & (ok.TAXITIME_SEC_mvt < hi)
    if m.sum() == 0: continue
    a, b = ok.loc[m, "err_adsb"].dropna(), ok.loc[m, "err_aobt3"].dropna()
    print(f"[{lo:>5},{hi:>7.0f})   {int(m.sum()):>6,} {np.sqrt((a**2).mean()):>12.1f} "
          f"{np.sqrt((b**2).mean()) if len(b) else float('nan'):>12.1f} "
          f"{a.median():>10.1f} {b.median() if len(b) else float('nan'):>11.1f}")
