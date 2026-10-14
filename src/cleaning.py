"""Loading and cleaning for the PRC 2026 movement data.

Every rule here is grounded in a measurement on the delivered data rather than a generic
heuristic; the numbers quoted in the comments come from the reconnaissance pass over all
12 training files (4,167,797 movements, 2,085,047 of them DEP).

Three findings shape this module:

1. **The target is exactly ``MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt``** (100.000% exact
   match on train DEP rows). ``BLOCK_TIME_UTC_mvt`` is **100% blank in ranking.parquet**,
   while ``MVT_TIME_UTC_mvt`` (actual takeoff) is 100% populated there. So the task is
   really "given when it took off, infer when it left the stand", and anything derived
   from the off-block time is unusable at prediction time. See :data:`INFERENCE_BLIND`.

2. **A tiny tail dominates an RMSE metric.** Against a per-airport-median baseline the
   584 DEP rows over 2h (0.028% of rows) carry 46% of all squared error, and the 4,126
   rows over 1h (0.198%) carry 55%. Leaving them in the training set drags the fit; they
   are therefore excluded from fitting via :func:`trainable_mask`, not from the data.

3. **The tail has a crisp signature: a missing ``AIRCRAFT_TYPE_mvt``.** 75% of DEP rows
   over 2h have no aircraft type, versus 0.07% of rows overall — a ~1000x enrichment,
   almost all at LIRF. Those rows are a genuinely different regime (n=1,507,
   mean 6,451s / std 11,170s, against 987s / 432s for normal rows) and ``ranking.parquet``
   contains 108 of them, so they must still be *predicted*. They are flagged rather than
   dropped so the modelling step can treat them as their own regime.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .airports import DEPARTURE_AIRPORTS

DATA_DIR = Path("Dataset")

#: The scored set. The organizers' final phase (announced 2026-10-08) replaced the Jan/Jul
#: `ranking.parquet` with `final_ranking.parquet`: Jan, Feb, Jun and Jul 2026, 670,790 DEP rows,
#: a strict superset of the 344,841 earlier IDs. Set ``PRC_RANKING=ranking`` to rebuild against
#: the earlier leaderboard file instead.
RANKING_STEM = os.environ.get("PRC_RANKING", "final_ranking")
SUBMITTING_STEM = RANKING_STEM.replace("ranking", "submitting")


def ranking_path(data_dir: Path | str = DATA_DIR) -> Path:
    """Path of the ranking file every ranking-side feature build must read."""
    return Path(data_dir) / f"{RANKING_STEM}.parquet"


def submitting_path(data_dir: Path | str = DATA_DIR) -> Path:
    """Path of the submission template matching :func:`ranking_path`."""
    return Path(data_dir) / f"{SUBMITTING_STEM}.parquet"

#: Columns that exist in the training files but are blank in ``ranking.parquet``.
#: Using any of these — or anything derived from them — as a model feature produces a
#: model that cannot be applied to the ranking set. ``TAXITIME_SEC_mvt`` is the target.
INFERENCE_BLIND = ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]

#: Training-set plausibility ceiling, in seconds. Above this a "taxi time" is almost
#: always a date/recording error (the largest values sit at ~24h and ~36h, and where the
#: independent ADS-B off-block time is available it disagrees by >1h for 96 of 103 such
#: rows). 2h is well beyond any real taxi-out: it is the 99.997th percentile.
MAX_PLAUSIBLE_TAXI_SEC = 7200

#: Tolerance for calling two independent off-block estimates "the same event", in seconds.
#: 30 min is wide enough to absorb ordinary plan-vs-actual slippage and narrow enough that a
#: whole-record date shift (hours) never looks like agreement.
OFFBLOCK_AGREE_SEC = 1800

#: Values in ``RUNWAY_mvt`` that mean "unknown" rather than naming a runway.
RUNWAY_NULL_TOKENS = {"NA", "N/A", "", "UNKNOWN"}


def load_training(data_dir: Path | str = DATA_DIR, add_source_month: bool = True) -> pd.DataFrame:
    """Load and concatenate the 12 monthly training files."""
    data_dir = Path(data_dir)
    files = sorted(glob.glob(str(data_dir / "training_*.parquet")))
    if not files:
        raise FileNotFoundError(f"no training_*.parquet under {data_dir}")

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        if add_source_month:
            df["source_month"] = Path(f).name.split("training_")[1].split("_")[0]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_ranking(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load ``ranking.parquet`` (ARR rows labelled, DEP rows are what we must predict)."""
    return pd.read_parquet(ranking_path(data_dir))


def load_submission_template(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load ``submitting.parquet`` — the exact ID set our submission must reproduce."""
    return pd.read_parquet(submitting_path(data_dir))


def clean_movements(df: pd.DataFrame, phase: str | None = "DEP") -> pd.DataFrame:
    """Apply the cleaning rules and attach data-quality flags.

    The function is deliberately **non-destructive except for provably-wrong values**: it
    repairs what is certainly an artefact, flags what is merely suspicious, and leaves the
    decision about excluding rows to :func:`trainable_mask`. That matters because the same
    function must run over ``ranking.parquet``, where nothing may be dropped.
    """
    out = df.copy()

    if phase is not None:
        out = out[out["PHASE_mvt"] == phase].copy()

    # --- 1. Negative taxi times ------------------------------------------------------
    # 369 DEP rows, every one between -12s and -1s, concentrated at LSZH (298) and LFPG
    # (61). The magnitudes make these clock/rounding artefacts, not real events, so they
    # are clipped to zero rather than dropped. Flag retained for auditing.
    if "TAXITIME_SEC_mvt" in out.columns:
        neg = out["TAXITIME_SEC_mvt"] < 0
        out["dq_was_negative"] = neg.fillna(False)
        out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].clip(lower=0)

    # --- 2. "NA" runway token --------------------------------------------------------
    # RUNWAY_mvt uses the literal string "NA" for unknown (8 rows at EDDF, and similar
    # elsewhere). Left as a string it silently becomes its own category.
    if "RUNWAY_mvt" in out.columns:
        out["RUNWAY_mvt"] = out["RUNWAY_mvt"].where(
            ~out["RUNWAY_mvt"].isin(RUNWAY_NULL_TOKENS), other=pd.NA
        )

    # --- 3. The missing-aircraft-type regime -----------------------------------------
    # See module docstring: this is the signature of the pathological long-taxi cluster.
    out["dq_no_aircraft_type"] = out["AIRCRAFT_TYPE_mvt"].isna()

    # --- 4. Implausible target -------------------------------------------------------
    if "TAXITIME_SEC_mvt" in out.columns:
        out["dq_implausible_target"] = (
            out["TAXITIME_SEC_mvt"] > MAX_PLAUSIBLE_TAXI_SEC
        ).fillna(False)

    # --- 5. Movement/flight-plan disagreement ----------------------------------------
    # The organizers state these inconsistencies are intentional and unreconciled, so we
    # encode the disagreement as a feature instead of "fixing" either side. Agreement
    # rates on train DEP: ADEP 100%, ADES 98.59%, aircraft type 99.66%, flight rule 99.93%.
    out["dq_ades_mismatch"] = _disagree(out, "ADES_mvt", "ADES_flt")
    out["dq_actype_mismatch"] = _disagree(out, "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt")
    out["dq_rule_mismatch"] = _disagree(out, "FLIGHT_RULE_mvt", "FLIGHT_RULE_flt")
    # a filed destination different from the actual one indicates a diversion
    out["dq_diverted"] = _disagree(out, "ADES_flt", "ADES_FILED_flt")

    # --- 6. Flight-plan record matched at all? ---------------------------------------
    # All 19 `_flt` columns share a single missingness pattern (1.08% of train DEP,
    # 1.48% of ranking DEP) — it is one "no NM flight matched" event, not 19 independent
    # gaps, so it collapses to one flag.
    out["dq_no_flight_record"] = out["FLIGHT_ID_mvt"].isna()

    # --- 7. Airport scope -------------------------------------------------------------
    out["dq_offscope_airport"] = ~out["ADEP_mvt"].isin(DEPARTURE_AIRPORTS)

    return out


def _disagree(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    """True where both columns are present and differ (NaN on either side -> False)."""
    if a not in df.columns or b not in df.columns:
        return pd.Series(False, index=df.index)
    both = df[a].notna() & df[b].notna()
    return (both & (df[a] != df[b])).fillna(False)


def trainable_mask(df: pd.DataFrame) -> pd.Series:
    """Rows suitable for fitting the main regressor.

    Excludes only rows whose *label* cannot be trusted — implausible targets and the
    missing-aircraft-type regime. Nothing is excluded on feature grounds, and this must
    never be applied to ``ranking.parquet``: every ranking row has to be predicted.
    """
    mask = pd.Series(True, index=df.index)
    if "dq_implausible_target" in df.columns:
        mask &= ~df["dq_implausible_target"]
    if "dq_no_aircraft_type" in df.columns:
        mask &= ~df["dq_no_aircraft_type"]
    if "TAXITIME_SEC_mvt" in df.columns:
        mask &= df["TAXITIME_SEC_mvt"].notna()
    return mask


def add_offblock_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Add taxi-time proxies from the off-block timestamps that survive into ranking.

    ``AOBT_3_flt`` is the actual off-block time derived from the flown (ADS-B) trajectory.
    It is *not* the airport's block time, so it is not the target — but it is available in
    98.5% of ranking DEP rows and, used raw as a prediction, already achieves RMSE ~385s
    against a target whose standard deviation is 546s. It is by a wide margin the single
    most informative field in the dataset, and its bias is strongly airport-specific
    (LTFM +236s, LIRF -214s), so let the model learn that correction per airport.

    The planned off-block times (``EOBT_1_flt`` / ``IOBT_flt`` / ``LOBT_flt``) give weaker
    but still useful proxies (RMSE ~677s / ~740s / ~740s).
    """
    out = df.copy()
    takeoff = out["MVT_TIME_UTC_mvt"]
    for col, name in [
        ("AOBT_3_flt", "proxy_taxi_aobt3"),     # actual off-block, flown trajectory
        ("EOBT_1_flt", "proxy_taxi_eobt1"),     # estimated off-block, filed plan
        ("IOBT_flt", "proxy_taxi_iobt"),        # initial off-block
        ("LOBT_flt", "proxy_taxi_lobt"),        # last off-block
        ("SCHED_TIME_UTC_mvt", "proxy_taxi_sched"),
    ]:
        if col in out.columns:
            out[name] = (takeoff - out[col]).dt.total_seconds()

    # --- how much the independent off-block estimates corroborate each other -------------
    # The four plan-side off-block times are measured by a different system from AOBT_3, so
    # their agreement with it says how far the ADS-B proxy can be trusted on this row. It
    # separates two cases a single proxy value cannot: a record whose date is wrong (every
    # timestamp shifts together, so the proxy is right) from one where AOBT_3 alone is bad.
    # Measured on training DEP rows: where all four agree, the proxy lands within 15 min of
    # the target 86.3% of the time; where none do, 60.2%.
    proxy_cols = [c for c in ["proxy_taxi_sched", "proxy_taxi_eobt1", "proxy_taxi_iobt",
                              "proxy_taxi_lobt"] if c in out.columns]
    if proxy_cols and "proxy_taxi_aobt3" in out.columns:
        delta = out[proxy_cols].sub(out["proxy_taxi_aobt3"], axis=0).abs()
        out["offblock_agree_n"] = (delta < OFFBLOCK_AGREE_SEC).sum(axis=1)
        out["offblock_agree_min"] = delta.min(axis=1)
        out["offblock_plan_median"] = out[proxy_cols].median(axis=1)
        out["offblock_plan_spread"] = out[proxy_cols].max(axis=1) - out[proxy_cols].min(axis=1)
        # the proxy read against the plan-side consensus, which is what the override keys on
        out["proxy_vs_plan"] = out["proxy_taxi_aobt3"] - out["offblock_plan_median"]

    # How far the actual departure ran from plan — a congestion/disruption signal that
    # uses only fields present in the ranking set.
    if {"AOBT_3_flt", "EOBT_1_flt"} <= set(out.columns):
        out["offblock_delay_sec"] = (
            out["AOBT_3_flt"] - out["EOBT_1_flt"]
        ).dt.total_seconds()
    if {"MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"} <= set(out.columns):
        out["takeoff_vs_sched_sec"] = (
            out["MVT_TIME_UTC_mvt"] - out["SCHED_TIME_UTC_mvt"]
        ).dt.total_seconds()

    return out


def cleaning_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Summarise what cleaning changed and how many rows each flag caught."""
    rows = [("input rows", len(raw), 1.0), ("DEP rows kept", len(cleaned), len(cleaned) / len(raw))]
    for col in [c for c in cleaned.columns if c.startswith("dq_")]:
        n = int(cleaned[col].sum())
        rows.append((col, n, n / len(cleaned)))
    if "TAXITIME_SEC_mvt" in cleaned.columns and cleaned["TAXITIME_SEC_mvt"].notna().any():
        n = int(trainable_mask(cleaned).sum())
        rows.append(("trainable rows", n, n / len(cleaned)))
    return pd.DataFrame(rows, columns=["item", "n", "share"])
