"""Runway configuration and stand rotation.

Two feature families measured in round 4 (`SKILLS.md` §11) and worth **-1.36 s full /
-1.86 s trimmed** on the mirrored protocol. Both were identified by `experiments/e18_gaps.py`
in round 2 and never built.

**Why they are not already covered.** The shipped features say how much traffic surrounds a
flight (congestion counts) and how slowly the surface is running (rolling realised durations).
Neither says **which physical layout the airport is working**. Two departures from the same
stand to the same runway travel different routes depending on which *other* runways are
active — crossings, backtracks and holding points all change — and that is a first-order
driver of taxi distance.

**Availability.** Every input is 100% present in `ranking.parquet`: `MVT_TIME_UTC_mvt`,
`RUNWAY_mvt`, `STAND_mvt`, plus the ARR rows' `TAXITIME_SEC_mvt`, which are labelled on both
sides. The ranking file is a complete movement record (`docs/findings.md` finding 6), so the traffic
universe is observed at the same rate in training and at prediction time. Measured skew:
medians match within 1% on every column and only 0.06% of ranking rows sit on a departure
configuration never seen in training.

Symmetric (±W) windows are used for the configuration, matching the existing congestion
features — the surrounding traffic is observed in both files, so nothing here needs the
target or any field absent from the ranking set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import movement_airport

#: Half-width of the window used to read off the configuration in use, in minutes.
CONFIG_WINDOW_MIN = 30

#: How far back to look when deciding whether the configuration has just changed.
CONFIG_LAG_MIN = 60

#: Columns needed from the raw movement files to build the traffic universe.
UNIVERSE_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
                 "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt"]


def _epoch_ns(s: pd.Series) -> np.ndarray:
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")


def prepare_universe(movements: pd.DataFrame) -> pd.DataFrame:
    """Resolve the airport each movement happened at and normalise the runway column.

    ``RUNWAY_mvt`` uses a literal ``"NA"`` string for unknown (see `docs/data_dictionary.md`),
    which would otherwise become a runway in its own right and pollute every configuration
    identifier it appears in.
    """
    out = movements.copy()
    out["apt"] = movement_airport(out)
    rwy = out["RUNWAY_mvt"].astype("string")
    out["rwy"] = rwy.where(rwy.ne("NA"), pd.NA)
    return out


def config_features(q_apt: pd.Series, q_ns: np.ndarray, q_rwy: pd.Series,
                    universe: pd.DataFrame, prefix: str,
                    window_min: int = CONFIG_WINDOW_MIN,
                    lag_min: int = CONFIG_LAG_MIN) -> pd.DataFrame:
    """Which runways the airport was working around each query time.

    Per query row: the configuration identifier (airport-namespaced, so ``EDDF|18+25C`` and
    ``LFPG|18+25C`` cannot collide), how many runways were in use, this flight's runway's share
    of the movements in the window, the window's movement count, and whether the configuration
    differed ``lag_min`` minutes earlier — a configuration change is itself a disruption event.

    Implemented with a cumulative per-runway count matrix so any window's runway multiset is
    two array lookups rather than a scan; each airport has at most a handful of runways, so the
    matrix is small.
    """
    W = int(window_min * 60 * 1e9)
    LAG = int(lag_min * 60 * 1e9)
    n = len(q_ns)
    out = {f"{prefix}{k}": v for k, v in {
        "cfg_id": np.full(n, "", dtype=object),
        "cfg_n_rwy": np.zeros(n, "float32"),
        "cfg_rwy_share": np.full(n, np.nan, "float32"),
        "cfg_changed": np.full(n, np.nan, "float32"),
        "cfg_n_mvt": np.zeros(n, "float32"),
    }.items()}

    u = universe[universe["rwy"].notna() & universe["MVT_TIME_UTC_mvt"].notna()]
    apt_arr = q_apt.to_numpy().astype(str)
    rwy_arr = q_rwy.astype("string").to_numpy()

    for apt, g in u.groupby("apt", observed=True, sort=False):
        qi = np.flatnonzero(apt_arr == apt)
        if len(qi) == 0:
            continue
        raw_ns = _epoch_ns(g["MVT_TIME_UTC_mvt"])
        order = np.argsort(raw_ns, kind="stable")
        t = raw_ns[order]
        codes, levels = pd.factorize(g["rwy"].to_numpy()[order])
        R = len(levels)
        C = np.zeros((len(t) + 1, R), dtype="int32")
        C[np.arange(1, len(t) + 1), codes] = 1
        np.cumsum(C, axis=0, out=C)

        def window_counts(qq: np.ndarray) -> np.ndarray:
            lo = np.searchsorted(t, qq - W, "left")
            hi = np.searchsorted(t, qq + W, "right")
            return C[hi] - C[lo]

        qq = q_ns[qi]
        cnt = window_counts(qq)
        present = cnt > 0
        tot = cnt.sum(1)
        out[f"{prefix}cfg_n_rwy"][qi] = present.sum(1)
        out[f"{prefix}cfg_n_mvt"][qi] = tot

        lvl = np.asarray(levels, dtype=object)
        ids = np.empty(len(qi), dtype=object)
        uniq, inv = np.unique(present, axis=0, return_inverse=True)
        names = [f"{apt}|" + "+".join(sorted(row_lvl)) if len(row_lvl) else f"{apt}|none"
                 for row_lvl in (lvl[row] for row in uniq)]
        for k, nm in enumerate(names):
            ids[inv == k] = nm
        out[f"{prefix}cfg_id"][qi] = ids

        own = pd.Index(levels).get_indexer(rwy_arr[qi])
        ok = own >= 0
        share = np.full(len(qi), np.nan, "float32")
        idx = np.flatnonzero(ok)
        share[ok] = np.where(tot[ok] > 0, cnt[idx, own[ok]] / np.maximum(tot[ok], 1), np.nan)
        out[f"{prefix}cfg_rwy_share"][qi] = share

        prev = window_counts(qq - LAG) > 0
        out[f"{prefix}cfg_changed"][qi] = (prev != present).any(1).astype("float32")

    return pd.DataFrame(out)


def rotation_features(q: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """How long the inbound that last used this stand had been parked on it.

    Arrival rows carry ``TAXITIME_SEC_mvt`` on both sides — they are labelled even in
    ``ranking.parquet`` — so in-block is ``MVT_TIME + TAXITIME``. The most recent in-block at
    the same (airport, stand) strictly before this departure's off-block anchor is the
    turnaround this aircraft has just completed.

    ``allow_exact_matches=False`` keeps the join strictly backward, so a departure can never
    see an arrival that had not yet finished taxiing in.
    """
    arr = universe[(universe["PHASE_mvt"] == "ARR") & universe["STAND_mvt"].notna()
                   & universe["TAXITIME_SEC_mvt"].notna()].copy()
    arr["inblock"] = arr["MVT_TIME_UTC_mvt"] + pd.to_timedelta(arr["TAXITIME_SEC_mvt"], unit="s")
    arr["stand_key"] = arr["apt"].astype(str) + "|" + arr["STAND_mvt"].astype(str)
    arr = (arr[["inblock", "stand_key", "TAXITIME_SEC_mvt"]]
           .rename(columns={"TAXITIME_SEC_mvt": "prev_arr_taxi"})
           .dropna(subset=["inblock"]).sort_values("inblock"))

    left = pd.DataFrame({
        "anchor": q["AOBT_3_flt"].fillna(q["SCHED_TIME_UTC_mvt"]),
        "stand_key": q["ADEP_mvt"].astype(str) + "|" + q["STAND_mvt"].astype(str),
    }, index=q.index).sort_values("anchor")
    keyed = left.dropna(subset=["anchor"])
    merged = pd.merge_asof(keyed, arr, left_on="anchor", right_on="inblock", by="stand_key",
                           direction="backward", allow_exact_matches=False)
    merged.index = keyed.index

    out = pd.DataFrame(index=q.index)
    turn = (merged["anchor"] - merged["inblock"]).dt.total_seconds()
    # a turnaround over 24 h means the join found an unrelated earlier occupant, not a rotation
    out["stand_turnaround_sec"] = turn.reindex(q.index).where(lambda s: s.between(0, 86400))
    out["stand_prev_arr_taxi"] = merged["prev_arr_taxi"].reindex(q.index)
    return out


def build_config_features(q: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    """Both families for one side, keyed by ``MVT_ID_mvt``.

    ``q`` is the DEP rows to build for; ``movements`` is the *unfiltered* frame from the same
    source, used as the traffic universe. Departure and arrival configurations are measured
    separately: the departure set fixes this flight's route, the arrival set fixes what it has
    to cross or give way to.
    """
    u = prepare_universe(movements)
    qn = _epoch_ns(q["MVT_TIME_UTC_mvt"])
    dep = config_features(q["ADEP_mvt"], qn, q["RUNWAY_mvt"], u[u["PHASE_mvt"] == "DEP"], "depcfg_")
    arr = config_features(q["ADEP_mvt"], qn, q["RUNWAY_mvt"], u[u["PHASE_mvt"] == "ARR"], "arrcfg_")
    cfg = pd.concat([dep, arr], axis=1)
    cfg.index = q.index
    return pd.concat([q[["MVT_ID_mvt"]], cfg, rotation_features(q, u)], axis=1)


#: The two configuration identifiers are high-cardinality strings and must be handed to
#: LightGBM as native categoricals; `model_features` only auto-detects numeric columns.
CONFIG_CATEGORICALS = ["depcfg_cfg_id", "arrcfg_cfg_id"]
