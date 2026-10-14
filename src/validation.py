"""Validation protocol for the PRC 2026 taxi-out model.

The contest scores Jan-2026 and Jul-2026 — an **out-of-time forecast**. Measured on the
current ``ranking.parquet`` (the 2026-09-04 revision, 344,841 DEP rows):

  * January is **44.3%** of the scored rows (152,719), July **55.7%** (192,122).
  * **All ten departure airports appear in both months.**

.. warning::
   An earlier revision of ``ranking.parquet`` had only 215,876 DEP rows, was 70.7% January,
   and contained *only* EDDF/EGLL/EHAM in July. The organizers replaced the file on
   2026-09-04 and the old one is missing 128,965 required rows — a submission built from it
   is rejected outright ("Submission is missing 128965 required rows present in truth").
   If any analysis quotes a 70/30 split or a three-airport July, it predates that revision.
   :func:`ranking_composition` always reads the weights from the file, so it self-corrects.

A uniform holdout over 2025 still scores a distribution the leaderboard does not look at, so
this module builds a split that mirrors the real one and reports the error two ways.

**The design tension, stated plainly.** Mirroring the *season* and staying *forward in time*
cannot both be fully satisfied with one year of training data: the only January in the
labelled data is January 2025, which sits at the very start, so validating on it means
training on months that come after it. The compromise here:

  * **train on Jan-Jun 2025**, validate on **Jul 2025** (the July slice, restricted to the
    three airports actually scored) and **Dec 2025** (standing in for January: the nearest
    held-out winter month, all ten airports).
  * Both validation months lie strictly *after* the training window, so this is a genuine
    forward backtest rather than interpolation.

The costs are real and should be quoted alongside any number this produces: the model fits on
six months rather than twelve, which understates what the final all-data model will do; and
December is a proxy for January, with atypical holiday traffic of its own. Use
:func:`naive_time_split` for the plain "train early, test late" comparison — the two numbers
answer different questions and it is worth reporting both.

**Rows are never dropped from the validation set.** ``trainable_mask`` filters what the model
*fits* on, not what it is *scored* on: the ranking set contains 108 rows of the anomalous
missing-aircraft-type regime and the leaderboard scores every one of them. Excluding them
from validation would flatter the model in exactly the way that matters least.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .airports import DEPARTURE_AIRPORTS

#: Airports scored in the July slice. As of the 2026-09-04 ranking revision this is all ten
#: departure airports; the previous revision had only EDDF/EGLL/EHAM, and hard-coding that
#: subset now silently drops 7/10 of the July slice from validation.
JULY_AIRPORTS = list(DEPARTURE_AIRPORTS)

#: Default backtest windows, as calendar months of 2025.
TRAIN_MONTHS = [1, 2, 3, 4, 5, 6]
VAL_WINTER_MONTH = 12   # stands in for January
VAL_SUMMER_MONTH = 7


def ranking_composition(ranking: pd.DataFrame) -> pd.DataFrame:
    """The (slice, airport) weights the leaderboard actually applies.

    Returns one row per scored cell with its share of the ranking DEP rows, so a validation
    score can be re-weighted onto the real composition instead of its own.
    """
    dep = ranking[ranking["PHASE_mvt"] == "DEP"]
    month = dep["MVT_TIME_UTC_mvt"].dt.month
    comp = (pd.DataFrame({
                "slice": np.where(month == 7, "jul", "jan"),
                "airport": dep["ADEP_mvt"].to_numpy(),
            })
            .value_counts(["slice", "airport"])
            .rename("n").reset_index())
    comp["weight"] = comp["n"] / comp["n"].sum()
    return comp.sort_values(["slice", "airport"], ignore_index=True)


def mirrored_split(
    df: pd.DataFrame,
    train_months: list[int] | None = None,
    winter_month: int = VAL_WINTER_MONTH,
    summer_month: int = VAL_SUMMER_MONTH,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split into (train, validation) masks plus the validation cell labels.

    The third return value labels each validation row ``"jan"`` or ``"jul"`` so
    :func:`weighted_rmse` can re-weight the cells onto the ranking composition.
    """
    train_months = TRAIN_MONTHS if train_months is None else train_months
    month = df["MVT_TIME_UTC_mvt"].dt.month

    is_train = month.isin(train_months)
    is_winter = month == winter_month
    is_summer = (month == summer_month) & df["ADEP_mvt"].isin(JULY_AIRPORTS)
    is_val = is_winter | is_summer

    cell = pd.Series(pd.NA, index=df.index, dtype="string")
    cell[is_winter] = "jan"
    cell[is_summer] = "jul"
    return is_train, is_val, cell


def weighted_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slices: pd.Series,
    airports: pd.Series,
    composition: pd.DataFrame,
) -> float:
    """RMSE re-weighted so each (slice, airport) cell carries its leaderboard share.

    Computed as ``sqrt(sum_c w_c * MSE_c)`` rather than a flat mean, because the validation
    set's own cell proportions are an accident of the calendar and are not what is scored.
    Cells absent from the validation data are renormalised away rather than assumed zero.
    """
    err = pd.DataFrame({
        "slice": slices.to_numpy(), "airport": airports.to_numpy(),
        "sq": (np.asarray(y_pred) - np.asarray(y_true)) ** 2,
    })
    mse = err.groupby(["slice", "airport"], observed=True)["sq"].mean().rename("mse").reset_index()
    m = mse.merge(composition[["slice", "airport", "weight"]], on=["slice", "airport"], how="inner")
    if m.empty:
        raise ValueError("no validation cell overlaps the ranking composition")
    w = m["weight"] / m["weight"].sum()
    return float(np.sqrt((w * m["mse"]).sum()))


def report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    frame: pd.DataFrame,
    composition: pd.DataFrame,
    slices: pd.Series,
    trimmed_mask: pd.Series | None = None,
) -> dict:
    """Both numbers, plus the per-airport breakdown.

    ``rmse_full`` is what the leaderboard computes — every row, anomalies included.
    ``rmse_trimmed`` excludes the pathological regime and is the number to compare model
    variants on: a change worth 5 s of real RMSE is invisible under the noise of a hundred
    rows whose targets run to 24 hours.
    """
    airports = frame["ADEP_mvt"]
    out = {
        "n_val": int(len(y_true)),
        "rmse_full": weighted_rmse(y_true, y_pred, slices, airports, composition),
        "rmse_unweighted": float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))),
    }
    if trimmed_mask is not None:
        tm = trimmed_mask.to_numpy()
        out["n_trimmed"] = int(tm.sum())
        out["rmse_trimmed"] = weighted_rmse(
            np.asarray(y_true)[tm], np.asarray(y_pred)[tm], slices[tm], airports[tm], composition
        )

    per = pd.DataFrame({
        "slice": slices.to_numpy(), "airport": airports.to_numpy(),
        "err": np.asarray(y_pred) - np.asarray(y_true),
    })
    out["per_airport"] = (per.groupby(["slice", "airport"], observed=True)["err"]
                             .agg(n="size", bias="mean", rmse=lambda e: float(np.sqrt((e ** 2).mean())))
                             .reset_index()
                             .merge(composition[["slice", "airport", "weight"]],
                                    on=["slice", "airport"], how="left")
                             .sort_values("weight", ascending=False, ignore_index=True))
    return out


def naive_time_split(df: pd.DataFrame, last_train_month: int = 10) -> tuple[pd.Series, pd.Series]:
    """The plain "train early, validate late" split, for comparison only.

    Reported alongside the mirrored number to show how much of any difference is composition
    rather than modelling. It is not the protocol.
    """
    month = df["MVT_TIME_UTC_mvt"].dt.month
    return month <= last_train_month, month > last_train_month
