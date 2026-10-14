#!/usr/bin/env python
"""Render data/processed/feature_catalogue.csv into docs/features.md."""
from pathlib import Path
import pandas as pd

c = pd.read_csv("data/processed/feature_catalogue.csv")
DESC = {
 "proxy_taxi_aobt3":"`MVT_TIME − AOBT_3_flt`. Takeoff minus the ADS-B-derived actual off-block. The single best estimator of the target.",
 "proxy_taxi_eobt1":"`MVT_TIME − EOBT_1_flt`. Same, against the filed estimated off-block.",
 "proxy_taxi_iobt":"`MVT_TIME − IOBT_flt` (initial off-block).",
 "proxy_taxi_lobt":"`MVT_TIME − LOBT_flt` (last off-block).",
 "proxy_taxi_sched":"`MVT_TIME − SCHED_TIME_UTC_mvt`. Absorbs the whole departure delay, which is why it drives the regime model.",
 "offblock_delay_sec":"`AOBT_3_flt − EOBT_1_flt`. How far the actual push ran from plan.",
 "takeoff_vs_sched_sec":"`MVT_TIME − SCHED_TIME`. Lateness at takeoff.",
 "offblock_agree_n":"How many of the four plan-side off-block estimates sit within 30 min of `AOBT_3_flt` (0–4).",
 "offblock_agree_min":"Smallest absolute gap between `AOBT_3_flt` and any plan-side estimate.",
 "offblock_plan_median":"Median of the four plan-side proxies — a consensus off-block independent of ADS-B.",
 "offblock_plan_spread":"Max − min across the four plan-side proxies. Disagreement among the plan timestamps.",
 "proxy_vs_plan":"`proxy_taxi_aobt3 − offblock_plan_median`. ADS-B against the plan consensus.",
 "stand_area":"Airport-namespaced apron/pier group parsed from `STAND_mvt` (letter prefix, else the hundreds/tens block).",
 "stand_full":"Airport-namespaced full stand identifier.",
 "stand_number":"Numeric part of the stand.",
 "surf_deptaxi_ratio":"This flight's `proxy_taxi_aobt3` divided by the airport's rolling 60-min mean. Airport-free measure of 'slower than normal right now'.",
 "sec_since_prev_dep_rwy":"Seconds since the previous departure on the same runway.",
 "sec_to_next_dep_rwy":"Seconds to the next departure on the same runway.",
 "queue_at_pushback":"Aircraft already taxiing out when this one pushed back.",
 "queue_at_takeoff":"Aircraft still taxiing out when this one got airborne.",
 "planned_enroute_sec":"`ARVT_1_flt − EOBT_1_flt`. Plan-side haul length; both endpoints are filed pre-departure.",
 "RUNWAY_mvt":"Departure runway actually used. Sets the taxi destination, so it pairs with the stand.",
 "ADEP_mvt":"Departure airport (10 in scope). Carries each airport's own baseline taxi level.",
 "ADES_mvt":"Destination airport. Stands in for haul length, terminal allocation and departure direction.",
 "AIRCRAFT_OPERATOR_flt":"Anonymised (hashed) airline. Encodes handler, stand allocation and pushback practice.",
 "AIRCRAFT_TYPE_mvt":"ICAO aircraft type from the movement record. Also the regime flag: null marks the anomalous population.",
 "AIRCRAFT_TYPE_flt":"ICAO aircraft type from the flight plan; differs from the movement value on 0.34% of rows.",
 "MARKET_SEGMENT_flt":"Mainline / Regional / Low-Cost / Business / Cargo / Charter / Military.",
 "WK_TBL_CAT_flt":"Wake turbulence category (L/M/H/J). Drives departure separation.",
 "dest_region":"First letter of the destination ICAO — continent-scale direction.",
 "dest_country":"First two letters of the destination ICAO.",
 "is_longhaul":"`planned_enroute_sec > 6 h`.",
 "is_domestic_region":"Destination shares the departure airport's ICAO first letter.",
 "local_minute_of_day":"Minutes since local midnight — the departure bank position.",
 "stand_rwy":"(not shipped) apron x runway as a raw categorical; measured neutral, see SKILLS.md 8.",
}
def desc(r):
    f = r.feature
    if f in DESC: return DESC[f]
    if f.startswith("surf_arrtaxi"):  base="Rolling mean **arrival** taxi-in at this airport"
    elif f.startswith("surf_deptaxi"):base="Rolling mean realised taxi-out at this airport"
    elif f.startswith("surf_depdelay"):base="Rolling mean `AOBT_3 − EOBT_1` at this airport"
    elif f.startswith("surf_rwy"):    base="Rolling mean realised taxi-out, same runway"
    elif f.startswith("surf_arearwy"):base="Rolling mean realised taxi-out, same apron → same runway"
    elif f.startswith("surf_area"):   base="Rolling mean realised taxi-out, same apron"
    elif f.startswith("surf_dep_over_arr"): return "Ratio of rolling departure taxi-out to arrival taxi-in."
    elif f.startswith("cong_dep"):    return "Departures at this airport within ±W min of takeoff."
    elif f.startswith("cong_arr"):    return "Arrivals at this airport within ±W min of takeoff."
    elif f.startswith("cong_rwy"):    return "Departures from this runway within ±W min of takeoff."
    elif f.startswith("cong_all"):    return "All movements at this airport within ±W min."
    elif f.startswith("wx_"):         return "Open-Meteo hourly value joined on (airport, hour), or a threshold flag derived from one."
    elif f.startswith("wind_") or f.startswith("gust_"): return "Wind decomposed against the runway heading parsed from `RUNWAY_mvt`."
    elif f.startswith("dq_"):         return "Data-quality flag set during cleaning."
    else: return "Raw delivered column, or a direct calendar part of the local civil time."
    if f.endswith(("_30","_60","_180")):
        w = f.rsplit("_",1)[1]
        return f"{base} over the preceding {w} min (backward, half-open — a flight never sees itself)."
    return base + "."
c["what it is"] = c.apply(desc, axis=1)

lines = ["# Model features — the complete inventory",
 "", "Auto-generated by `experiments/feature_catalogue.py`; regenerate after any feature change.",
 "", "**127 model inputs** (112 numeric, 15 categorical) feeding the LightGBM regressor described in",
 "`SKILLS.md` §6. `gain %` is the share of total split gain in the shipped all-12-month booster",
 "(`data/processed/lgbm_full.txt`) — the model that produced the v3 submission.",
 "", "Every feature obeys the availability rule from `src/cleaning.py`: it must be computable identically",
 "on `ranking.parquet`. `rank %` is its non-null rate there; a large gap against `train %` would be a",
 "train/serve skew. None exceeds 0.8%.", "",
 "## Summary by family", "",
 "| family | features | gain % | what it measures |", "|---|---|---|---|"]
WHAT = {
 "off-block proxies":"Takeoff minus each available off-block estimate. The target is an off-block time, so these are near-direct measurements of it.",
 "stand / apron":"Where the aircraft started. The best available proxy for taxi distance.",
 "raw movement & flight-plan fields":"Delivered columns used as-is: runway, airline, aircraft type, airport.",
 "surface state (rolling realised durations)":"How long taxis are *actually* taking right now, from movements already completed.",
 "off-block corroboration":"How far the independent off-block estimates agree — i.e. how much to trust the proxy.",
 "route / destination":"Where the flight is going, and how far.",
 "weather":"Open-Meteo hourly conditions plus runway-relative wind components.",
 "congestion (counts, +/-W min)":"How many other movements surround this one.",
 "time (local civil)":"Calendar position in the airport's own local time.",
 "runway sequence":"Spacing to the neighbouring departures on the same runway.",
 "queue (aircraft on the surface)":"How many aircraft are physically taxiing out at the same instant.",
 "holidays":"Public-holiday calendar for the airport's country/subdivision.",
 "data-quality flags":"Cleaning flags, retained so the model can branch on record quality.",
}
fam = c.groupby("family").agg(n=("feature","size"), g=("gain_%","sum")).sort_values("g", ascending=False)
for f, r in fam.iterrows():
    lines.append(f"| {f} | {int(r.n)} | **{r.g:.1f}** | {WHAT.get(f,'')} |")
lines += ["", "## Every feature, by importance", "",
 "| # | feature | family | kind | gain % | splits | train % | rank % | what it is |", "|---|---|---|---|---|---|---|---|---|"]
for i, r in c.iterrows():
    rk = "—" if pd.isna(r["rank_present_%"]) else f"{r['rank_present_%']:.1f}"
    kind = r["kind"] + (f" ({int(r.n_unique)})" if not pd.isna(r.n_unique) else "")
    lines.append(f"| {i+1} | `{r.feature}` | {r.family} | {kind} | {r['gain_%']:.3f} | {r.splits:,} | "
                 f"{r['train_present_%']:.1f} | {rk} | {r['what it is']} |")
Path("docs").mkdir(exist_ok=True)
Path("docs/features.md").write_text("\n".join(lines) + "\n")
print(f"wrote docs/features.md ({len(c)} features)")
