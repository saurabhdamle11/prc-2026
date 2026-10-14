#!/usr/bin/env python
"""E26: gate the shipped delta model with the placeholder classifier, rather than replace it.

E25 established two facts that point in opposite directions:

* Among the 1,507 regime rows, **47.9% carry a placeholder `BLOCK_TIME_UTC_mvt` equal to the
  scheduled time**, and for those `proxy_taxi_sched` predicts the target with an RMSE of
  **4 seconds**. They are also the *long* rows -- median target 7,534 s against 1,030 s for
  the rest -- so they are exactly the rows carrying the squared error.
* A mixture that replaces the branch entirely is **worse** than the shipped delta model, and
  so is the same mixture with the mode known exactly (4,505 s vs 4,240 s). Predicting the
  non-placeholder mode by its mean delta throws away what the delta model already does well.

So the estimator E25 should have tested is neither: keep the shipped delta model for the
non-placeholder mode and use `proxy_taxi_sched` for the placeholder mode, gated on the
classifier (AUC 0.75-0.94 out of sample). This measures that, against its own oracle, and
reports the delta model's error split by mode -- which is the quantity that says whether any
gain is available at all.
"""
from __future__ import annotations
import sys
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import model_features
from src.postprocess import apply_regime_model, fit_regime_model

PLACEHOLDER_TOL = 60
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
R["placeholder"] = ((R.BLOCK_TIME_UTC_mvt - R.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()
                    <= PLACEHOLDER_TOL)
usable = [c for c in feats if R[c].notna().mean() > 0.5]

recs, per_mode = [], []
for mth in range(1, 13):
    te, trm = R[R.month == mth], R[R.month != mth]
    if len(te) < 10:
        continue
    yt = te.TAXITIME_SEC_mvt.to_numpy()
    px = te.proxy_taxi_sched.to_numpy(dtype="float64")
    ph = te.placeholder.to_numpy()

    m = fit_regime_model(d[d.month != mth], feats)
    base, _ = apply_regime_model(te, np.full(len(te), np.nan), m)

    clf = lgb.LGBMClassifier(random_state=0, n_jobs=-1, **CLF_PARAMS)
    clf.fit(trm[usable], trm.placeholder.astype(int))
    p = clf.predict_proba(te[usable])[:, 1]
    fin = np.isfinite(px)

    cand = {"delta model (shipped)": base,
            "soft blend p*proxy+(1-p)*delta": np.where(fin, p * px + (1 - p) * base, base),
            "ORACLE gate": np.where(ph & fin, px, base)}
    for thr in (0.5, 0.7, 0.9):
        cand[f"hard gate p>{thr}"] = np.where((p > thr) & fin, px, base)

    recs.append({"month": mth, "n": len(te),
                 **{k: np.sqrt(np.nanmean((v - yt) ** 2)) for k, v in cand.items()}})
    per_mode.append({"month": mth, "n_ph": int(ph.sum()), "n_other": int((~ph).sum()),
                     "delta_rmse_ph": np.sqrt(np.nanmean((base[ph] - yt[ph]) ** 2)),
                     "delta_rmse_other": np.sqrt(np.nanmean((base[~ph] - yt[~ph]) ** 2)),
                     "p_mean_ph": p[ph].mean(), "p_mean_other": p[~ph].mean()})

t = pd.DataFrame(recs).set_index("month"); n = t.pop("n")
pm = pd.DataFrame(per_mode).set_index("month")

print("=== the shipped delta model's error, split by mode ===")
print("If it already predicts the placeholder rows well, no gate can help.\n")
print(pm.round(2).to_string())
w_ph = pm.n_ph / (pm.n_ph + pm.n_other)
print(f"\ncount-weighted: placeholder rows {np.sqrt((pm.delta_rmse_ph**2 * pm.n_ph).sum()/pm.n_ph.sum()):,.0f}s"
      f"   other rows {np.sqrt((pm.delta_rmse_other**2 * pm.n_other).sum()/pm.n_other.sum()):,.0f}s")

print("\n=== leave-one-month-out RMSE on the regime rows ===")
print(t.round(0).to_string())
w = n / n.sum()
print("\ncount-weighted (the number that matters):")
print((t.pow(2).mul(w, axis=0).sum() ** 0.5).sort_values().round(1).to_string())
print("\nJuly alone:")
print(t.loc[7].sort_values().round(1).to_string())
