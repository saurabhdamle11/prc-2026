#!/usr/bin/env python
"""E20: runway configuration and stand rotation -- the two families `e18_gaps.py` identified
and that were never built.

Why these two, specifically. The shipped 127 features tell the model *how much* traffic
surrounds a flight (congestion counts, 1.2% of gain) and *how slowly* the surface is running
(rolling realised durations, 10.6%). Neither tells it **which physical layout the airport is
running**. A departure from the same stand to the same runway takes a different route
depending on which *other* runways are active -- crossings, backtracks and holding points all
change. That is a first-order driver of taxi distance and it is currently invisible.

Both families are computable identically on `ranking.parquet`: they need only
`MVT_TIME_UTC_mvt`, `RUNWAY_mvt` and `STAND_mvt`, all of which are 100% present there, plus
the labelled ARR rows (finding 6 in docs/findings.md -- the ranking file is a complete movement
record, so the traffic universe is observed at the same rate on both sides).

Symmetric (+/-W) windows are used for the configuration, matching the existing congestion
features and the availability rule: the surrounding traffic is observed in both files.
"""
from __future__ import annotations
import glob, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import movement_airport

RAW_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
            "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt"]


def _ns(s: pd.Series) -> np.ndarray:
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")


def load_universe(which: str) -> pd.DataFrame:
    """Every movement record on one side, with the airport it happened at resolved."""
    if which == "train":
        m = pd.concat([pd.read_parquet(f, columns=RAW_COLS)
                       for f in sorted(glob.glob("Dataset/training_*.parquet"))],
                      ignore_index=True)
    else:
        m = pd.read_parquet("Dataset/ranking.parquet", columns=RAW_COLS)
    m["apt"] = movement_airport(m)
    rwy = m["RUNWAY_mvt"].astype("string")
    m["rwy"] = rwy.where(rwy.ne("NA"), pd.NA)     # the file uses a literal "NA" for unknown
    return m


def config_features(q_apt: pd.Series, q_ns: np.ndarray, q_rwy: pd.Series,
                    universe: pd.DataFrame, prefix: str, window_min: int = 30,
                    lag_min: int = 60) -> pd.DataFrame:
    """Which runways the airport was working around each query time.

    Returns, per query row: the configuration identifier (airport-namespaced, so 'EDDF|18+25C'
    and 'LFPG|18+25C' never collide), how many runways were in use, this flight's runway's
    share of the movements in the window, and whether the configuration differed `lag_min`
    minutes earlier -- a configuration change is itself a surface-disruption event.
    """
    W = int(window_min * 60 * 1e9)
    LAG = int(lag_min * 60 * 1e9)
    n = len(q_ns)
    out = {"cfg_id": np.full(n, "", dtype=object), "cfg_n_rwy": np.zeros(n, "float32"),
           "cfg_rwy_share": np.full(n, np.nan, "float32"),
           "cfg_changed": np.full(n, np.nan, "float32"),
           "cfg_n_mvt": np.zeros(n, "float32")}
    out = {f"{prefix}{k}": v for k, v in out.items()}
    u = universe[universe["rwy"].notna() & universe["MVT_TIME_UTC_mvt"].notna()]
    q_apt = q_apt.to_numpy().astype(str)
    q_rwy = q_rwy.astype("string").to_numpy()

    for apt, g in u.groupby("apt", observed=True, sort=False):
        qi = np.flatnonzero(q_apt == apt)
        if len(qi) == 0:
            continue
        t = np.sort(_ns(g["MVT_TIME_UTC_mvt"]))
        order = np.argsort(_ns(g["MVT_TIME_UTC_mvt"]), kind="stable")
        codes, levels = pd.factorize(g["rwy"].to_numpy()[order])
        R = len(levels)
        # cumulative per-runway counts -> any window's runway multiset is two lookups
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
        # the configuration string: runways in use, sorted, airport-namespaced
        ids = np.empty(len(qi), dtype=object)
        uniq, inv = np.unique(present, axis=0, return_inverse=True)
        names = [f"{apt}|" + "+".join(sorted(lvl[row])) if row.any() else f"{apt}|none"
                 for row in uniq]
        for k, nm in enumerate(names):
            ids[inv == k] = nm
        out[f"{prefix}cfg_id"][qi] = ids
        own = pd.Index(levels).get_indexer(q_rwy[qi])
        ok = own >= 0
        share = np.full(len(qi), np.nan, "float32")
        share[ok] = np.where(tot[ok] > 0, cnt[np.flatnonzero(ok), own[ok]] / np.maximum(tot[ok], 1), np.nan)
        out[f"{prefix}cfg_rwy_share"][qi] = share
        prev = window_counts(qq - LAG) > 0
        out[f"{prefix}cfg_changed"][qi] = (prev != present).any(1).astype("float32")
    return pd.DataFrame(out)


def rotation_features(q: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """How long the inbound that last used this stand had been parked on it.

    The arrival rows carry `TAXITIME_SEC_mvt` on both sides (they are labelled even in
    `ranking.parquet`), so in-block = `MVT_TIME + TAXITIME`. The most recent in-block at the
    same (airport, stand) strictly before this departure's off-block anchor is the turnaround
    this aircraft has just completed.
    """
    arr = universe[(universe.PHASE_mvt == "ARR") & universe.STAND_mvt.notna()
                   & universe.TAXITIME_SEC_mvt.notna()].copy()
    arr["inblock"] = arr["MVT_TIME_UTC_mvt"] + pd.to_timedelta(arr["TAXITIME_SEC_mvt"], unit="s")
    arr = (arr[["apt", "STAND_mvt", "inblock", "TAXITIME_SEC_mvt"]]
           .rename(columns={"TAXITIME_SEC_mvt": "prev_arr_taxi"})
           .dropna(subset=["inblock"]).sort_values("inblock"))
    arr["stand_key"] = arr["apt"].astype(str) + "|" + arr["STAND_mvt"].astype(str)

    left = pd.DataFrame({
        "anchor": q["AOBT_3_flt"].fillna(q["SCHED_TIME_UTC_mvt"]),
        "stand_key": q["ADEP_mvt"].astype(str) + "|" + q["STAND_mvt"].astype(str),
    }, index=q.index).sort_values("anchor")
    merged = pd.merge_asof(
        left.dropna(subset=["anchor"]), arr[["inblock", "stand_key", "prev_arr_taxi"]],
        left_on="anchor", right_on="inblock", by="stand_key", direction="backward",
        allow_exact_matches=False)
    merged.index = left.dropna(subset=["anchor"]).index
    turn = (merged["anchor"] - merged["inblock"]).dt.total_seconds()
    out = pd.DataFrame(index=q.index)
    out["stand_turnaround_sec"] = turn.reindex(q.index).where(lambda s: s.between(0, 86400))
    out["stand_prev_arr_taxi"] = merged["prev_arr_taxi"].reindex(q.index)
    return out


def build(side: str, q: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    u = load_universe(side)
    qn = _ns(q["MVT_TIME_UTC_mvt"])
    # departures and arrivals are separate configurations: the departure set fixes this
    # flight's route, the arrival set fixes what it has to cross or give way to.
    dep_cfg = config_features(q["ADEP_mvt"], qn, q["RUNWAY_mvt"],
                              u[u.PHASE_mvt == "DEP"], "depcfg_")
    arr_cfg = config_features(q["ADEP_mvt"], qn, q["RUNWAY_mvt"],
                              u[u.PHASE_mvt == "ARR"], "arrcfg_")
    cfg = pd.concat([dep_cfg, arr_cfg], axis=1)
    cfg.index = q.index
    rot = rotation_features(q, u)
    out = pd.concat([q[["MVT_ID_mvt"]], cfg, rot], axis=1)
    print(f"  {side}: built in {time.time()-t0:.0f}s  "
          f"depcfg levels {out.depcfg_cfg_id.nunique()}  "
          f"arrcfg levels {out.arrcfg_cfg_id.nunique()}  "
          f"turnaround matched {out.stand_turnaround_sec.notna().mean():.1%}", flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    for side, src in [("train", "data/processed/train_features.parquet"),
                      ("ranking", "data/processed/ranking_features.parquet")]:
        if which not in ("both", side):
            continue
        q = pd.read_parquet(src, columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                                          "RUNWAY_mvt", "STAND_mvt", "AOBT_3_flt",
                                          "SCHED_TIME_UTC_mvt"])
        build(side, q).to_parquet(f"data/processed/{side}_config.parquet", index=False)
        print(f"  -> data/processed/{side}_config.parquet")
