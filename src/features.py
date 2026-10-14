"""Feature engineering for the PRC 2026 taxi-out model.

Everything here obeys the availability rule established in :mod:`src.cleaning`: a feature is
usable only if it can be computed identically on ``ranking.parquet``. The one disqualifying
column is ``BLOCK_TIME_UTC_mvt`` (100% blank there) — the target is exactly
``MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt``, so the off-block time *is* the thing being
predicted. ``MVT_TIME_UTC_mvt`` (actual takeoff) is 100% populated in the ranking set and is
therefore a legitimate anchor, which is what makes the congestion family below possible.

**Why the congestion features are valid.** They count *other* movements around each flight,
so they are only meaningful if the ranking file is a complete record of traffic in its
months rather than a sample. Measured: daily movement counts per airport in Jan-2026 match
Jan-2025 within +/-5% (EHAM is the one outlier at 0.88, consistent with Schiphol's real
capacity cap), the DEP/ARR balance is 50/50 in both files, and all 31 days of each month are
present. Counts are therefore on the same scale in training and at prediction time.

**The traffic universe matters.** Congestion is computed against *all* movements at the
airport, both phases — arrivals compete for taxiways and runway crossings. So these
functions take a separate ``movements`` frame (the full unfiltered file) alongside the DEP
rows being featurised. Pass the training concatenation when featurising training rows and
``ranking.parquet`` when featurising ranking rows; never mix the two, and never use training
traffic to describe a 2026 flight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Minute windows for the +/- congestion counts.
DEFAULT_WINDOWS = (15, 30, 60)

_NS_PER_MIN = 60_000_000_000


def movement_airport(df: pd.DataFrame) -> pd.Series:
    """The airport a movement physically happened at.

    ``ADEP_mvt`` on DEP rows, ``ADES_mvt`` on ARR rows. Getting this wrong is the classic
    silent error in this dataset: filtering on ``ADEP_mvt`` alone mixes departures from an
    airport with arrivals *from* it, attributing one airport's runways and stands to every
    airport it flies to. See ``docs/data_dictionary.md``.
    """
    return pd.Series(
        np.where(df["PHASE_mvt"] == "DEP", df["ADEP_mvt"], df["ADES_mvt"]),
        index=df.index,
    )


def _epoch_ns(s: pd.Series) -> np.ndarray:
    """UTC timestamps -> int64 nanoseconds, with NaT as NaN-safe sentinel handling upstream."""
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")


def _sorted_by_airport(movements: pd.DataFrame, col: str, phase: str | None = None) -> dict[str, np.ndarray]:
    """Per-airport sorted arrays of one timestamp column, for searchsorted lookups."""
    m = movements
    if phase is not None:
        m = m[m["PHASE_mvt"] == phase]
    apt = movement_airport(m)
    t = m[col]
    ok = t.notna() & apt.notna()
    apt, t = apt[ok], t[ok]
    ns = _epoch_ns(t)
    out: dict[str, np.ndarray] = {}
    for icao, idx in pd.Series(np.arange(len(apt)), index=apt.to_numpy()).groupby(level=0):
        out[icao] = np.sort(ns[idx.to_numpy()])
    return out


def _window_count(sorted_ns: np.ndarray, query_ns: np.ndarray, window_min: int) -> np.ndarray:
    """How many entries of ``sorted_ns`` fall within +/- ``window_min`` of each query."""
    half = window_min * _NS_PER_MIN
    lo = np.searchsorted(sorted_ns, query_ns - half, side="left")
    hi = np.searchsorted(sorted_ns, query_ns + half, side="right")
    return hi - lo


def add_congestion_features(
    df: pd.DataFrame,
    movements: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Counts of surrounding traffic, anchored on each row's actual takeoff time.

    Adds, for each window W in ``windows``:
      ``cong_dep_{W}``  departures at the same airport within +/-W minutes (excluding self)
      ``cong_arr_{W}``  arrivals at the same airport within +/-W minutes
      ``cong_all_{W}``  both phases
      ``cong_rwy_{W}``  departures from the same runway within +/-W minutes (excluding self)

    This is the "airport procedures and load" driver the organizers name, built entirely
    from fields present in both files.
    """
    out = df.copy()
    apt = movement_airport(out)
    q = _epoch_ns(out["MVT_TIME_UTC_mvt"])

    dep_idx = _sorted_by_airport(movements, "MVT_TIME_UTC_mvt", phase="DEP")
    arr_idx = _sorted_by_airport(movements, "MVT_TIME_UTC_mvt", phase="ARR")

    # same-runway index keys on (airport, runway)
    mv = movements[movements["PHASE_mvt"] == "DEP"]
    mv_apt = movement_airport(mv)
    rwy_key = mv_apt.astype(str) + "|" + mv["RUNWAY_mvt"].astype(str)
    ok = mv["MVT_TIME_UTC_mvt"].notna()
    rwy_ns = _epoch_ns(mv.loc[ok, "MVT_TIME_UTC_mvt"])
    rwy_idx: dict[str, np.ndarray] = {}
    for key, idx in pd.Series(np.arange(ok.sum()), index=rwy_key[ok].to_numpy()).groupby(level=0):
        rwy_idx[key] = np.sort(rwy_ns[idx.to_numpy()])

    q_rwy = apt.astype(str) + "|" + out["RUNWAY_mvt"].astype(str)

    for W in windows:
        for name, index, keys, drop_self in [
            (f"cong_dep_{W}", dep_idx, apt, True),
            (f"cong_arr_{W}", arr_idx, apt, False),
            (f"cong_rwy_{W}", rwy_idx, q_rwy, True),
        ]:
            vals = np.zeros(len(out), dtype="int32")
            for key, positions in pd.Series(np.arange(len(out)), index=keys.to_numpy()).groupby(level=0):
                arr = index.get(key)
                pos = positions.to_numpy()
                if arr is None or len(arr) == 0:
                    continue
                vals[pos] = _window_count(arr, q[pos], W)
            if drop_self:
                vals = np.maximum(vals - 1, 0)
            out[name] = vals
        out[f"cong_all_{W}"] = out[f"cong_dep_{W}"] + out[f"cong_arr_{W}"]

    return out


def add_queue_features(df: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    """Departure-queue length, the closest thing in this data to a physical queue.

    A flight is "on the surface taxiing out" between its actual off-block time
    (``AOBT_3_flt``, the ADS-B-derived pushback) and its takeoff (``MVT_TIME_UTC_mvt``).
    Counting how many such flights exist at a given instant gives the real congestion the
    aircraft is queueing behind, rather than a proxy count of nearby movements.

    Both timestamps survive into ``ranking.parquet`` (``AOBT_3_flt`` on 98.52% of DEP rows),
    so this is computable at prediction time. Adds:

      ``queue_at_pushback``  flights already taxiing out when this one pushed back
      ``queue_at_takeoff``   flights still taxiing out when this one got airborne

    Rows without an ``AOBT_3_flt`` get NaN for ``queue_at_pushback`` rather than a guessed
    pushback time — a wrong anchor is worse than an explicit gap the model can branch on.
    """
    out = df.copy()

    mv = movements[movements["PHASE_mvt"] == "DEP"]
    mv_apt = movement_airport(mv)
    ok = mv["AOBT_3_flt"].notna() & mv["MVT_TIME_UTC_mvt"].notna() & mv_apt.notna()
    push = pd.Series(_epoch_ns(mv.loc[ok, "AOBT_3_flt"]), index=mv_apt[ok].to_numpy())
    tko = pd.Series(_epoch_ns(mv.loc[ok, "MVT_TIME_UTC_mvt"]), index=mv_apt[ok].to_numpy())

    push_idx = {k: np.sort(v.to_numpy()) for k, v in push.groupby(level=0)}
    tko_idx = {k: np.sort(v.to_numpy()) for k, v in tko.groupby(level=0)}

    apt = movement_airport(out)
    for name, col in [("queue_at_pushback", "AOBT_3_flt"), ("queue_at_takeoff", "MVT_TIME_UTC_mvt")]:
        vals = np.full(len(out), np.nan)
        have = out[col].notna().to_numpy()
        q = np.zeros(len(out), dtype="int64")
        q[have] = _epoch_ns(out.loc[have, col])
        for key, positions in pd.Series(np.arange(len(out)), index=apt.to_numpy()).groupby(level=0):
            if key not in push_idx:
                continue
            pos = positions.to_numpy()
            pos = pos[have[pos]]
            if len(pos) == 0:
                continue
            # already pushed back, not yet airborne  ->  currently taxiing
            started = np.searchsorted(push_idx[key], q[pos], side="right")
            finished = np.searchsorted(tko_idx[key], q[pos], side="right")
            vals[pos] = started - finished
        out[name] = vals

    # each flight counts itself while it is on the surface
    out["queue_at_pushback"] = (out["queue_at_pushback"] - 1).clip(lower=0)
    out["queue_at_takeoff"] = (out["queue_at_takeoff"] - 1).clip(lower=0)
    return out


def add_sequence_features(df: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    """Spacing to the neighbouring departures on the same runway.

    Runway occupancy is serial, so the gap to the aircraft ahead is a direct measure of
    how hard the departure stream is being pushed.
    """
    out = df.copy()
    apt = movement_airport(out)
    key = apt.astype(str) + "|" + out["RUNWAY_mvt"].astype(str)
    q = _epoch_ns(out["MVT_TIME_UTC_mvt"])

    mv = movements[movements["PHASE_mvt"] == "DEP"]
    mv_apt = movement_airport(mv)
    mk = mv_apt.astype(str) + "|" + mv["RUNWAY_mvt"].astype(str)
    ok = mv["MVT_TIME_UTC_mvt"].notna()
    ns = _epoch_ns(mv.loc[ok, "MVT_TIME_UTC_mvt"])
    idx = {k: np.sort(ns[v.to_numpy()]) for k, v in
           pd.Series(np.arange(ok.sum()), index=mk[ok].to_numpy()).groupby(level=0)}

    prev = np.full(len(out), np.nan)
    nxt = np.full(len(out), np.nan)
    for k, positions in pd.Series(np.arange(len(out)), index=key.to_numpy()).groupby(level=0):
        arr = idx.get(k)
        if arr is None or len(arr) < 2:
            continue
        pos = positions.to_numpy()
        loc = np.searchsorted(arr, q[pos], side="left")
        has_prev = loc > 0
        prev[pos[has_prev]] = (q[pos[has_prev]] - arr[loc[has_prev] - 1]) / 1e9
        after = np.searchsorted(arr, q[pos], side="right")
        has_next = after < len(arr)
        nxt[pos[has_next]] = (arr[after[has_next]] - q[pos[has_next]]) / 1e9

    out["sec_since_prev_dep_rwy"] = prev
    out["sec_to_next_dep_rwy"] = nxt
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features in the airport's **local** civil time.

    Local, not UTC: an 07:00 local departure bank is a rush hour at every airport, but that
    is a different UTC hour at Istanbul than at Heathrow. ``local_time`` is produced by
    :func:`src.external_data.attach_holidays`, which must run first.
    """
    out = df.copy()
    if "local_time" not in out.columns:
        raise ValueError("run attach_holidays() first — local_time is required")

    lt = out["local_time"]
    out["local_minute_of_day"] = lt.dt.hour * 60 + lt.dt.minute
    out["local_month"] = lt.dt.month
    out["local_doy"] = lt.dt.dayofyear
    out["is_weekend"] = lt.dt.dayofweek >= 5

    for name, series, period in [
        ("tod", out["local_minute_of_day"], 1440),
        ("dow", lt.dt.dayofweek, 7),
        ("doy", out["local_doy"], 365.25),
    ]:
        ang = 2 * np.pi * series / period
        out[f"{name}_sin"], out[f"{name}_cos"] = np.sin(ang), np.cos(ang)
    return out


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Where the flight is going, as far as the data supports without an external gazetteer.

    ``ARVT_1_flt - EOBT_1_flt`` is the **planned** en-route duration: both endpoints come
    from the filed flight plan, so it is known before pushback and is a clean proxy for haul
    length. This is a deliberate, narrow use of ``ARVT_1_flt``, which ``SKILLS.md`` otherwise
    lists as a leakage trap — the trap is using it as an *absolute* time or using the flown
    ``ARVT_3_flt``, both of which encode what happened after departure. Only the plan-side
    difference is used here.
    """
    out = df.copy()
    if {"ARVT_1_flt", "EOBT_1_flt"} <= set(out.columns):
        out["planned_enroute_sec"] = (out["ARVT_1_flt"] - out["EOBT_1_flt"]).dt.total_seconds()
        out["is_longhaul"] = out["planned_enroute_sec"] > 6 * 3600

    dest = out["ADES_mvt"].astype(str)
    out["dest_region"] = dest.str[0]           # ICAO first letter: E/L Europe, K US, Z China...
    out["dest_country"] = dest.str[:2]
    out["is_domestic_region"] = out["dest_region"] == movement_airport(out).astype(str).str[0]

    return out


def add_stand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the parking stand into an apron/pier grouping.

    Stand is the best available proxy for distance from gate to runway threshold, but the
    naming is airport-local and takes two incompatible forms:

      * **letter-prefixed** at EDDF, EHAM and LFPG (``B41``, ``A62B``, ``V119``, ``F233``)
      * **purely numeric** at EGLL, EDDM, LEBL and LIRF (``583``, ``328``, ``247``, ``409``),
        where the leading digit is the terminal/apron block — EGLL ``5xx`` is Terminal 5
      * **mixed** at LEMD (70% numeric, plus ``T``-prefixed terminal stands) and at LSZH/LTFM

    So the grouping is the alpha prefix where one exists and the hundreds/tens block
    otherwise, always namespaced by airport because the numbering is only locally meaningful.
    A single regex for the letter form silently drops 44.5% of rows to null.
    """
    out = df.copy()
    if "STAND_mvt" not in out.columns:
        return out

    stand = out["STAND_mvt"].astype("string")
    apt = movement_airport(out).astype("string")

    alpha = stand.str.extract(r"^([A-Za-z]+)", expand=False)
    # float64 (not the nullable Float64) so NA compares as NaN instead of raising
    digits = pd.to_numeric(stand.str.extract(r"^(\d+)", expand=False), errors="coerce").astype("float64")
    block = np.where(np.isnan(digits), np.nan, np.where(digits >= 100, digits // 100, digits // 10))
    block = pd.Series(block, index=out.index)
    block = block.map(lambda v: pd.NA if pd.isna(v) else str(int(v))).astype("string")

    area = alpha.fillna(block)
    out["stand_area"] = (apt + "|" + area).astype("string")
    out["stand_full"] = (apt + "|" + stand).astype("string")
    out["stand_number"] = pd.to_numeric(stand.str.extract(r"(\d+)", expand=False), errors="coerce")
    return out


def runway_heading(runway: pd.Series) -> pd.Series:
    """Magnetic heading in degrees from a runway designator ('25C' -> 250, '07L' -> 70)."""
    num = runway.astype("string").str.extract(r"^(\d{1,2})", expand=False)
    return pd.to_numeric(num, errors="coerce") * 10.0


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """Head/cross-wind relative to the runway in use.

    Wind matters for taxi-out mainly through the runway configuration it forces, and a raw
    wind direction cannot express that — the same 270 degrees is a headwind on 27 and a
    tailwind on 09. Decomposing against the runway heading makes the effect learnable.

    Runway designators are *magnetic* while the forecast direction is *true*; European
    magnetic variation is a few degrees and is ignored here.
    """
    out = df.copy()
    need = {"RUNWAY_mvt", "wx_wind_direction_10m", "wx_wind_speed_10m"}
    if not need <= set(out.columns):
        return out
    hdg = runway_heading(out["RUNWAY_mvt"])
    delta = np.radians(out["wx_wind_direction_10m"] - hdg)
    out["wind_headwind_ms"] = out["wx_wind_speed_10m"] * np.cos(delta)
    out["wind_crosswind_ms"] = (out["wx_wind_speed_10m"] * np.sin(delta)).abs()
    if "wx_wind_gusts_10m" in out.columns:
        out["gust_crosswind_ms"] = (out["wx_wind_gusts_10m"] * np.sin(delta)).abs()
    return out


def add_weather_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Threshold flags for the weather regimes measured to matter in ``analysis.ipynb`` §12.2.

    Each threshold comes from the controlled (airport x hour x month) comparison, not a raw
    marginal — the raw visibility marginal is inverted by confounding with light-traffic
    winter nights.
    """
    out = df.copy()
    if "wx_visibility" in out.columns:
        out["wx_lvp"] = out["wx_visibility"] < 1000          # low-visibility procedures
        out["wx_vis_very_low"] = out["wx_visibility"] < 200
    if {"wx_temperature_2m", "wx_precipitation"} <= set(out.columns):
        out["wx_deice_regime"] = (out["wx_temperature_2m"] < 3) & (out["wx_precipitation"] > 0)
    if "wx_snowfall" in out.columns:
        out["wx_snowing"] = out["wx_snowfall"] > 0
    if "wx_wind_gusts_10m" in out.columns:
        out["wx_strong_gusts"] = out["wx_wind_gusts_10m"] > 15
    return out


def build_features(
    df: pd.DataFrame,
    movements: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Run the full feature pipeline over already-cleaned, already-joined DEP rows.

    Expects ``df`` to have been through ``clean_movements`` -> ``add_offblock_proxies`` ->
    ``attach_weather`` -> ``attach_holidays``. ``movements`` is the *unfiltered* frame from
    the same source file(s), used as the traffic universe for the congestion counts.
    """
    out = add_congestion_features(df, movements, windows=windows)
    out = add_queue_features(out, movements)
    out = add_sequence_features(out, movements)
    out = add_time_features(out)
    out = add_route_features(out)
    out = add_stand_features(out)
    out = add_wind_components(out)
    out = add_weather_regimes(out)
    return out


# --------------------------------------------------------------------------------------
# Feature selection for modelling
# --------------------------------------------------------------------------------------

#: Never feed these to a model. ``BLOCK_TIME_UTC_mvt`` and ``TAXITIME_SEC_mvt`` are the
#: target and its other half; ``ARVT_3_flt`` is the *flown* arrival, i.e. after the event;
#: the rest are identifiers or raw strings superseded by an engineered column.
EXCLUDE_FROM_MODEL = {
    "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt",     # target and its complement
    "ARVT_3_flt",                                  # post-event arrival — genuine leakage
    # Both of these are computed FROM the target, so they are unavailable at prediction time
    # and would leak the answer straight into the fit. `dq_no_aircraft_type` is different:
    # it reads AIRCRAFT_TYPE_mvt, which is populated in the ranking set, so it stays.
    "dq_implausible_target", "dq_was_negative",
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt",   # identifiers
    "PHASE_mvt", "source_month", "local_time", "local_date", "holiday_name",
    "STAND_mvt",                                   # superseded by stand_full / stand_area
    "ADEP_flt", "ADES_FILED_flt",                  # captured by the dq_* mismatch flags
    "wx_grid_lat", "wx_grid_lon", "wx_grid_elevation_m",   # weather grid bookkeeping
}

#: Columns to hand to LightGBM/CatBoost as native categoricals.
CATEGORICAL_FEATURES = [
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt",
    "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_RULE_mvt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "stand_full", "stand_area", "dest_region",
    "dest_country",
]


def model_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split the built frame into (numeric, categorical) model inputs.

    Raw ``datetime64`` columns are dropped: every one of them has been turned into a
    duration or a calendar part already, and feeding an absolute timestamp to a tree lets it
    memorise the training period instead of learning a pattern that transfers to 2026.
    """
    cats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    nums = [
        c for c in df.columns
        if c not in EXCLUDE_FROM_MODEL
        and c not in cats
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
    ]
    return nums, cats
