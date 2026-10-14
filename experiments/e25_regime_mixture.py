#!/usr/bin/env python
"""E25: the regime is a two-component mixture, and round 2 tested the wrong mixture.

E24 established the structure. For the 1,507 missing-aircraft-type rows (1,487 of them at
LIRF), the quantity the branch has to predict is `BLOCK - SCHED`, and its distribution is
**bimodal with a point mass at zero**: median 5 s, mean 2,278 s, sd 4,993 s. About half these
records carry a `BLOCK_TIME_UTC_mvt` that is simply the scheduled time -- a placeholder, not a
measurement. For those rows

    taxi = MVT_TIME - BLOCK = MVT_TIME - SCHED = proxy_taxi_sched

**exactly**, and `proxy_taxi_sched` is available at prediction time. For the rest, the off-block
delay is large and genuinely uncertain.

`e6_mixture.py` tried a mixture on this split and found it worse on all six splits, with a
*mode-oracle* worse than the plain linear (7,302 s vs 4,180 s). That result is real but it
tested the wrong estimator: it predicted each mode's **mean**, which throws away
`proxy_taxi_sched`'s variation inside the placeholder mode -- exactly the variation that makes
the mode worth identifying. The correct form is

    pred = p(x) * proxy_taxi_sched  +  (1 - p(x)) * (proxy_taxi_sched - delta(x))

with `p` a classifier for the placeholder mode and `delta` fitted on the non-placeholder rows
only. Under squared error that is the conditional mean of a mixture, which is what the metric
wants.

Scored leave-one-month-out on the regime rows alone, count-weighted, plus July on its own
because July carries the scored exposure for this regime 4:1.
"""
from __future__ import annotations
import sys
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import model_features
from src.postprocess import _REGIME_PARAMS, apply_regime_model, fit_regime_model

PLACEHOLDER_TOL = 60          # seconds; `e5_regime_structure.py` used the same threshold
CLF_PARAMS = dict(objective="binary", num_leaves=8, min_child_samples=20, n_estimators=300,
                  learning_rate=0.05, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                  verbose=-1)

d = pd.read_parquet("data/processed/train_features.parquet")
d = d.merge(pd.read_parquet("data/processed/train_surface.parquet"),
            on="MVT_ID_mvt", how="left", validate="one_to_one")
nums, cats = model_features(d)
for c in cats:
    d[c] = d[c].astype("string").fillna("__NA__").astype("category")
feats = nums + cats

d["month"] = d.MVT_TIME_UTC_mvt.dt.month
R = d[d.dq_no_aircraft_type & d.TAXITIME_SEC_mvt.notna()].copy()
R["bs"] = (R.BLOCK_TIME_UTC_mvt - R.SCHED_TIME_UTC_mvt).dt.total_seconds()
R["placeholder"] = R.bs.abs() <= PLACEHOLDER_TOL
R["delta"] = R.proxy_taxi_sched - R.TAXITIME_SEC_mvt

print(f"=== {len(R):,} regime rows; placeholder mode = |BLOCK - SCHED| <= {PLACEHOLDER_TOL}s ===")
print(f"placeholder: {R.placeholder.mean():.1%} ({int(R.placeholder.sum()):,} rows)")
for nm, g in [("placeholder", R[R.placeholder]), ("other", R[~R.placeholder])]:
    err = (g.proxy_taxi_sched - g.TAXITIME_SEC_mvt)
    print(f"  {nm:<12} n={len(g):>5}  target median {g.TAXITIME_SEC_mvt.median():>8,.0f}s  "
          f"|proxy_sched - y| median {err.abs().median():>8,.0f}s  "
          f"rmse of proxy_sched {np.sqrt((err**2).mean()):>9,.0f}s")

usable = [c for c in feats if R[c].notna().mean() > 0.5]
print(f"\nregime-usable features: {len(usable)} of {len(feats)}")

rows = []
for mth in range(1, 13):
    te, trm = R[R.month == mth], R[R.month != mth]
    if len(te) < 10:
        continue
    yt = te.TAXITIME_SEC_mvt.to_numpy()
    px = te.proxy_taxi_sched.to_numpy(dtype="float64")
    fallback = trm.TAXITIME_SEC_mvt.mean()
    cand = {}

    # (a) the shipped delta model, refitted without the held-out month
    src = d[d.month != mth]
    m = fit_regime_model(src, feats)
    cand["delta model (shipped)"], _ = apply_regime_model(te, np.full(len(te), np.nan), m)

    # (b) raw proxy
    cand["proxy_taxi_sched raw"] = np.where(np.isfinite(px), px, fallback)

    # (c) the mixture, done properly
    clf = lgb.LGBMClassifier(random_state=0, n_jobs=-1, **CLF_PARAMS)
    clf.fit(trm[usable], trm.placeholder.astype(int))
    p = clf.predict_proba(te[usable])[:, 1]
    other = trm[~trm.placeholder]
    dreg = lgb.LGBMRegressor(random_state=0, n_jobs=-1, **_REGIME_PARAMS)
    ok = other.delta.notna()
    dreg.fit(other.loc[ok, usable], other.delta[ok])
    dhat = dreg.predict(te[usable])
    mix = p * px + (1 - p) * (px - dhat)
    cand["mixture (clf x proxy + delta)"] = np.where(np.isfinite(mix), mix, fallback)
    cand["mixture, clf AUC-check"] = np.nan   # placeholder, replaced below

    # (d) upper bound: the mode known exactly, each mode predicted correctly
    dmean_other = other.delta[ok].mean()
    orc = np.where(te.placeholder, px, px - dmean_other)
    cand["ORACLE mode (proxy | proxy-delta)"] = np.where(np.isfinite(orc), orc, fallback)

    from sklearn.metrics import roc_auc_score
    auc = (roc_auc_score(te.placeholder, p) if te.placeholder.nunique() > 1 else np.nan)
    cand.pop("mixture, clf AUC-check")
    rows.append({"month": mth, "n": len(te), "auc": auc,
                 **{k: np.sqrt(np.nanmean((v - yt) ** 2)) for k, v in cand.items()}})

t = pd.DataFrame(rows).set_index("month")
n, auc = t.pop("n"), t.pop("auc")
print("\n=== leave-one-month-out RMSE on the regime rows ===")
print(t.round(0).to_string())
print(f"\nclassifier AUC per month: {auc.round(3).to_dict()}")
w = n / n.sum()
print("\ncount-weighted RMSE (the number that matters):")
print((t.pow(2).mul(w, axis=0).sum() ** 0.5).sort_values().round(1).to_string())
print("\nJuly alone (4x the scored exposure of any other month for this regime):")
print(t.loc[7].sort_values().round(1).to_string())
