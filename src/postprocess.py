"""Post-processing rules applied after the regressor, before writing a submission.

These exist because the main model provably cannot learn them: the rows they handle are
excluded from the fit by ``trainable_mask``, so a tree never sees their targets and cannot
extrapolate to them.

Two rules live here. The first handles the missing-aircraft-type regime; the second
(:func:`apply_extreme_proxy_override`) handles rows whose ADS-B off-block proxy is implausibly
large *and* corroborated by the rest of the record.

**The missing-aircraft-type regime.** Rows with a null ``AIRCRAFT_TYPE_mvt`` are a separate
population (training mean 6,451 s, median 2,643 s, std 11,170 s, against ~990 s for normal
rows). The current ``ranking.parquet`` contains **388** of them, 276 of those at LIRF in July.

The decisive measurement: **every one of these rows also has no flight-plan record**, so
``AOBT_3_flt`` — the model's dominant feature — is missing on 100% of them. That is why the
regressor predicts ~1,600 s for rows whose true mean is over 6,000 s. But
``SCHED_TIME_UTC_mvt`` *is* populated, and ``proxy_taxi_sched = MVT_TIME - SCHED_TIME``
correlates with the target at **Pearson 0.896** within this group. Regressing on it instead of
substituting a constant is worth ~180 s of weighted RMSE:

===========================================  ==============
strategy                                     weighted RMSE
===========================================  ==============
no branch                                            538.2 s
constant training mean                               515.4 s
**linear on proxy_taxi_sched**                   **335.7 s**
perfect prediction (unreachable ceiling)             295.1 s
===========================================  ==============

Measured out-of-sample: the fit is estimated on training-window anomalies only and scored on
validation anomalies, where it achieves RMSE 4,467 s against the constant's 11,895 s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Fallback when ``proxy_taxi_sched`` is unavailable for an anomalous row.
_FALLBACK_COL = "TAXITIME_SEC_mvt"


def fit_regime_branch(train: pd.DataFrame) -> dict:
    """Fit the missing-aircraft-type rule on labelled training rows.

    Returns the coefficients plus the constant fallback, so the caller can log exactly what
    was applied — the challenge's reproducibility requirement makes an opaque rule useless.
    """
    a = train[train["dq_no_aircraft_type"] & train[_FALLBACK_COL].notna()]
    if len(a) < 30:
        raise ValueError(f"only {len(a)} anomalous training rows — refusing to fit a branch")
    usable = a[a["proxy_taxi_sched"].notna()]
    slope, intercept = np.polyfit(usable["proxy_taxi_sched"], usable[_FALLBACK_COL], 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "constant_fallback": float(a[_FALLBACK_COL].mean()),
        "n_fitted": int(len(usable)),
    }


def apply_regime_branch(df: pd.DataFrame, pred: np.ndarray, branch: dict,
                        floor: float = 0.0) -> tuple[np.ndarray, int]:
    """Override predictions for missing-aircraft-type rows. Returns (predictions, n_changed)."""
    out = np.asarray(pred, dtype="float64").copy()
    mask = df["dq_no_aircraft_type"].to_numpy()
    if not mask.any():
        return out, 0
    sched = df.loc[mask, "proxy_taxi_sched"].to_numpy(dtype="float64")
    vals = branch["slope"] * sched + branch["intercept"]
    vals = np.where(np.isfinite(sched), vals, branch["constant_fallback"])
    out[mask] = np.maximum(vals, floor)
    return out, int(mask.sum())


# --------------------------------------------------------------------------------------
# The corroborated-implausible-proxy override
# --------------------------------------------------------------------------------------

#: Above this, ``proxy_taxi_aobt3`` is not a credible taxi time — either the record's date is
#: wrong (and every timestamp on it is shifted together) or ``AOBT_3_flt`` alone is bad.
EXTREME_PROXY_SEC = 7200

#: How many of the four plan-side off-block estimates must corroborate the proxy before it is
#: believed. Two is where the training evidence turns decisive; see the table below.
MIN_AGREE = 2


def apply_extreme_proxy_override(df, pred, floor: float = 0.0):
    """Trust an implausibly large ADS-B proxy when the rest of the record corroborates it.

    A gradient-boosted tree can never predict outside the range of targets it was fitted on,
    so a flight whose true taxi time is 24 h is predicted at ~1,000 s no matter what its
    features say. That is usually right — such a proxy is normally just a bad ``AOBT_3_flt``.
    But when the scheduled, initial, last and estimated off-block times *all* sit at the same
    implausible offset, the record's date is shifted as a whole, ``BLOCK_TIME_UTC_mvt`` is
    shifted with it, and the proxy is the correct answer.

    19 training DEP rows have ``proxy_taxi_aobt3 > 7200``. The agreement test splits them
    cleanly: it fires on 7, whose targets are all genuinely large, and abstains on 12, whose
    targets are ordinary taxis of 877-6,591 s. **It is right in direction on all 19.**

    On the 7 it fires on:

    ====================================  ==========
    prediction                            RMSE
    ====================================  ==========
    a normal taxi (~1,000 s)                33,836 s
    the plan consensus alone                 1,761 s
    **the midpoint (shipped)**             **1,946 s**
    the proxy alone                          2,153 s
    ====================================  ==========

    What matters is the *gating*, not the value: every choice of value recovers >94% of the
    gain, and picking between them on seven rows would be fitting noise. The midpoint is used
    because the two sources are independent measurements of the same shifted off-block.

    Restricted to the extreme band on purpose. In ``[3600, 7200)`` the same agreement test
    fires on 589 training rows and the group mean beats the proxy there (1,301 s vs 1,844 s),
    so overriding would make those rows worse. Nor does it extend to rows *missing*
    ``AOBT_3_flt`` whose plan times look implausible: 4,314 such training rows have a median
    target of 1,076 s, i.e. they are merely delayed, not shifted, and there is no second
    source to corroborate them.

    The current ``ranking.parquet`` has **5** rows in the extreme band, of which this rule
    fires on **2** (one LFPG at 13,313 s with all four estimates agreeing, one LIRF at
    86,340 s with three). It is a small, high-variance bet -- but the training evidence
    behind it is 19 for 19.

    Returns ``(predictions, n_changed)``.
    """
    out = np.asarray(pred, dtype="float64").copy()
    need = {"proxy_taxi_aobt3", "offblock_agree_n", "offblock_plan_median"}
    if not need <= set(df.columns):
        return out, 0
    proxy = df["proxy_taxi_aobt3"].to_numpy(dtype="float64")
    mask = (proxy > EXTREME_PROXY_SEC) & (df["offblock_agree_n"].to_numpy() >= MIN_AGREE)
    if not mask.any():
        return out, 0
    # the median of the corroborating estimates rather than the proxy alone: they are four
    # independent measurements of the same shifted off-block, so their centre is the better
    # estimate of it, and a single bad one cannot carry the row.
    plan = df["offblock_plan_median"].to_numpy(dtype="float64")
    vals = np.where(np.isfinite(plan), (proxy + plan) / 2.0, proxy)
    out[mask] = np.maximum(vals[mask], floor)
    return out, int(mask.sum())


# --------------------------------------------------------------------------------------
# The missing-aircraft-type regime, as a model rather than a line
# --------------------------------------------------------------------------------------

#: Hyperparameters for the regime model. Deliberately tiny — it is fitted on ~1,500 rows, so
#: anything with real capacity memorises them.
_REGIME_PARAMS = dict(objective="regression", num_leaves=8, min_child_samples=20,
                      n_estimators=300, learning_rate=0.05, colsample_bytree=0.7,
                      subsample=0.8, subsample_freq=1, verbose=-1)


def fit_regime_model(train, feature_cols, seed: int = 0) -> dict:
    """Fit the regime branch on ``proxy_taxi_sched - y`` instead of on ``y``.

    ``proxy_taxi_sched`` is known exactly at prediction time and explains most of the level of
    these targets (Pearson 0.896). Regressing ``y`` on it forces the model to re-derive that
    level; regressing the *difference* leaves it only the part that is actually uncertain, and
    lets the prediction follow the proxy out to values no tree could otherwise reach.

    Measured on six disjoint month splits, scoring the regime rows alone:

    ==========================  =========
    branch                      mean RMSE
    ==========================  =========
    linear on proxy_taxi_sched    4,180 s
    **LightGBM on the delta**   **4,039 s**
    ==========================  =========

    It wins on five of the six splits; the exception is a 89-row December holdout. July
    dominates the scored set for this regime (its anomaly rate is 4x January's), and the delta
    model wins there by 5-8%.
    """
    import lightgbm as lgb

    a = train[train["dq_no_aircraft_type"] & train[_FALLBACK_COL].notna()]
    if len(a) < 30:
        raise ValueError(f"only {len(a)} anomalous training rows — refusing to fit a branch")
    # these rows carry no NM flight record at all, so every _flt-derived column is null on
    # them; fitting on a column that is absent for the whole group just adds noise
    usable = [c for c in feature_cols if a[c].notna().mean() > 0.5]
    delta = a["proxy_taxi_sched"] - a[_FALLBACK_COL]
    ok = delta.notna()
    model = lgb.LGBMRegressor(random_state=seed, n_jobs=-1, **_REGIME_PARAMS)
    model.fit(a.loc[ok, usable], delta[ok])
    return {"model": model, "features": usable, "n_fitted": int(ok.sum()),
            "constant_fallback": float(a[_FALLBACK_COL].mean())}


def apply_regime_model(df, pred, branch: dict, floor: float = 0.0):
    """Override the missing-aircraft-type rows using :func:`fit_regime_model`."""
    out = np.asarray(pred, dtype="float64").copy()
    mask = df["dq_no_aircraft_type"].to_numpy()
    if not mask.any():
        return out, 0
    sub = df.loc[mask]
    vals = sub["proxy_taxi_sched"].to_numpy(dtype="float64") - branch["model"].predict(sub[branch["features"]])
    vals = np.where(np.isfinite(vals), vals, branch["constant_fallback"])
    out[mask] = np.maximum(vals, floor)
    return out, int(mask.sum())
