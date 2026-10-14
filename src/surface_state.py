"""Rolling airport-surface state features.

The existing congestion family counts *how many* movements surround a flight. It cannot say
how badly the surface is actually running: twenty departures an hour is a calm morning at
Heathrow and a meltdown at Zurich. These features measure the observed state directly, from
the most recent movements that had already happened when this aircraft took off.

**Why these are computable at prediction time.** Each one reads only fields that are fully
populated in ``ranking.parquet``:

* ``TAXITIME_SEC_mvt`` on **ARR** rows — arrivals are 100% labelled in the ranking file
  (344,693 rows), so recent taxi-*in* times are observable for the 2026 months exactly as
  they are in training. Taxi-in and taxi-out share taxiways, runway crossings and the same
  ground-control workload, so a slow arrival flow is direct evidence of a slow surface.
* ``MVT_TIME_UTC_mvt - AOBT_3_flt`` on **DEP** rows — the ADS-B off-block proxy, present on
  98.5% of ranking departures. Recent values give the realised taxi-out of aircraft that
  were airborne *before* this one.
* ``AOBT_3_flt - EOBT_1_flt`` on **DEP** rows — how far the departure bank is running behind
  its filed plan, i.e. the airport's current delay state.

**Strictly causal.** Every window is *backward-looking and half-open*: it uses movements with
``MVT_TIME < this row's MVT_TIME``, never at or after it. A symmetric window would let a
flight see its own outcome through its neighbours, and the anchor row is its own neighbour.
The shift is applied on the value array rather than by filtering, so ties at the same
timestamp resolve to "earlier in sort order only" and can never include self.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import movement_airport

#: Look-back windows in minutes.
DEFAULT_WINDOWS = (30, 60, 180)


def _as_ns(s: pd.Series) -> np.ndarray:
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")


def _rolling_backward(
    src_t: np.ndarray, src_v: np.ndarray, q_t: np.ndarray, window_min: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and count of ``src_v`` over ``[q_t - window, q_t)``, via a prefix sum.

    Both inputs must already be sorted by time. The half-open upper bound is what keeps the
    feature causal — ``side="left"`` excludes anything stamped at exactly the query time,
    which is where a flight would otherwise see itself.
    """
    csum = np.concatenate([[0.0], np.cumsum(src_v)])
    ccnt = np.arange(len(src_v) + 1, dtype="float64")
    lo = np.searchsorted(src_t, q_t - window_min * 60_000_000_000, side="left")
    hi = np.searchsorted(src_t, q_t, side="left")
    n = ccnt[hi] - ccnt[lo]
    total = csum[hi] - csum[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, total / n, np.nan)
    return mean, n


#: Realised taxi durations are winsorised before averaging. A single 24 h recording error in
#: the look-back window would otherwise move a 60-minute mean by tens of minutes and make the
#: feature actively misleading for every flight behind it.
CLIP_SEC = 7200


def add_keyed_surface_state(
    df: pd.DataFrame,
    movements: pd.DataFrame,
    key_cols: tuple[str, ...],
    name: str,
    windows: tuple[int, ...] = (60, 180),
) -> pd.DataFrame:
    """Backward-rolling mean realised taxi-out, grouped by a finer key than the airport.

    The airport-level features say how the surface is running overall. These say how it is
    running *for this route across it* -- the recent taxi times of departures from the same
    apron, or to the same runway. That is much closer to the quantity being predicted: the
    walk from stand to threshold is a fixed distance, and what varies is the queue on it.

    ``key_cols`` are columns that must exist on both ``df`` and ``movements``.
    """
    out = df.copy()
    mv = movements[movements["PHASE_mvt"] == "DEP"]
    vals = (mv["MVT_TIME_UTC_mvt"] - mv["AOBT_3_flt"]).dt.total_seconds().clip(0, CLIP_SEC)
    ok = vals.notna() & mv["MVT_TIME_UTC_mvt"].notna()

    # A missing key part becomes an explicit "__NA__" bucket rather than pd.NA: an NA inside
    # the key array makes np.lexsort raise, and silently dropping those rows would leave the
    # feature null for every flight off an unparsed stand.
    def _key(frame, base):
        k = base.fillna("__NA__")
        for c in key_cols:
            k = k + "|" + frame[c].astype("string").fillna("__NA__")
        return k

    s_key = _key(mv.loc[ok], movement_airport(mv).astype("string")[ok])
    q_key = _key(out, movement_airport(out).astype("string"))

    s_key = s_key.to_numpy(dtype=object)
    s_t, s_v = _as_ns(mv.loc[ok, "MVT_TIME_UTC_mvt"]), vals[ok].to_numpy("float64")
    order = np.lexsort((s_t, s_key))
    s_key, s_t, s_v = s_key[order], s_t[order], s_v[order]
    bounds = {k: (lo, hi) for k, lo, hi in _group_bounds(s_key)}

    q_t = _as_ns(out["MVT_TIME_UTC_mvt"])
    cols = {f"surf_{name}_mean_{W}": np.full(len(out), np.nan) for W in windows}
    cnts = {f"surf_{name}_n_{W}": np.zeros(len(out)) for W in windows}
    for k, positions in pd.Series(np.arange(len(out)), index=q_key.to_numpy(dtype=object)).groupby(level=0):
        b = bounds.get(k)
        if b is None:
            continue
        lo, hi = b
        pos = positions.to_numpy()
        for W in windows:
            m, n = _rolling_backward(s_t[lo:hi], s_v[lo:hi], q_t[pos], W)
            cols[f"surf_{name}_mean_{W}"][pos] = m
            cnts[f"surf_{name}_n_{W}"][pos] = n
    for k, v in {**cols, **cnts}.items():
        out[k] = v
    return out


def add_surface_state(
    df: pd.DataFrame,
    movements: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Attach the rolling-state features to ``df``, measured over ``movements``.

    ``movements`` is the unfiltered frame from the same source as ``df`` — the training
    concatenation when featurising training rows, ``ranking.parquet`` when featurising
    ranking rows. Mixing the two would describe a 2026 flight with 2025 traffic.
    """
    out = df.copy()
    q_apt = movement_airport(out).to_numpy()
    q_t_all = _as_ns(out["MVT_TIME_UTC_mvt"])

    mv = movements
    mv_apt = movement_airport(mv)

    # --- the three source signals, each a (airport -> sorted (t, v)) index ----------
    arr = mv[mv["PHASE_mvt"] == "ARR"]
    arr_ok = arr["TAXITIME_SEC_mvt"].notna() & arr["MVT_TIME_UTC_mvt"].notna()
    sources = {
        "arrtaxi": (mv_apt[arr.index[arr_ok]].to_numpy(),
                    _as_ns(arr.loc[arr_ok, "MVT_TIME_UTC_mvt"]),
                    arr.loc[arr_ok, "TAXITIME_SEC_mvt"].to_numpy("float64")),
    }

    dep = mv[mv["PHASE_mvt"] == "DEP"]
    prox = (dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds()
    dly = (dep["AOBT_3_flt"] - dep["EOBT_1_flt"]).dt.total_seconds()
    for name, vals in [("deptaxi", prox), ("depdelay", dly)]:
        ok = vals.notna() & dep["MVT_TIME_UTC_mvt"].notna()
        sources[name] = (mv_apt[dep.index[ok]].to_numpy(),
                         _as_ns(dep.loc[ok, "MVT_TIME_UTC_mvt"]),
                         vals[ok].to_numpy("float64"))

    for name, (s_apt, s_t, s_v) in sources.items():
        order = np.lexsort((s_t, s_apt))
        s_apt, s_t, s_v = s_apt[order], s_t[order], s_v[order]
        bounds = {k: (lo, hi) for k, lo, hi in _group_bounds(s_apt)}

        cols = {f"surf_{name}_mean_{W}": np.full(len(out), np.nan) for W in windows}
        cnts = {f"surf_{name}_n_{W}": np.zeros(len(out)) for W in windows}
        for key, positions in pd.Series(np.arange(len(out)), index=q_apt).groupby(level=0):
            b = bounds.get(key)
            if b is None:
                continue
            lo, hi = b
            pos = positions.to_numpy()
            # queries must be sorted for searchsorted-free prefix lookups to stay cheap
            qt = q_t_all[pos]
            for W in windows:
                m, n = _rolling_backward(s_t[lo:hi], s_v[lo:hi], qt, W)
                cols[f"surf_{name}_mean_{W}"][pos] = m
                cnts[f"surf_{name}_n_{W}"][pos] = n
        for k, v in {**cols, **cnts}.items():
            out[k] = v

    # A flight's own proxy read against what the surface is currently doing. The level of
    # `surf_deptaxi_mean_60` is airport-specific; the *ratio* is not, so it transfers.
    if "proxy_taxi_aobt3" in out.columns:
        base = out["surf_deptaxi_mean_60"]
        out["surf_deptaxi_ratio"] = out["proxy_taxi_aobt3"] / base.where(base > 0)
    for W in windows:
        a, b = out[f"surf_deptaxi_mean_{W}"], out[f"surf_arrtaxi_mean_{W}"]
        out[f"surf_dep_over_arr_{W}"] = a / b.where(b > 0)
    return out


def _group_bounds(sorted_keys: np.ndarray):
    """(key, start, stop) for each run of equal values in a sorted key array."""
    if len(sorted_keys) == 0:
        return
    change = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [len(sorted_keys)]])
    for s, e in zip(starts, stops):
        yield sorted_keys[s], int(s), int(e)
