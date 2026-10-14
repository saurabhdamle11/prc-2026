#!/usr/bin/env python
"""E24: the missing-aircraft-type regime is 25.6% of the metric on 0.12% of rows. What is it?

E19's re-measurement puts this population at 25.6% of the weighted MSE (round 2 had it at
23.4%, before the normal-row model improved and made it relatively larger). E21 localises it:
337 of the 428 validation rows sit in `jul LIRF`, where they carry 64% of the worst cell.
Oracle value of solving it: 279.8 -> 241.4s, a **38.4s** headroom -- by far the largest single
item left.

Under squared error the only thing that can be done with a population this noisy is to get its
**conditional mean** right, subgroup by subgroup. So the questions here are, in order:

1. What does the target distribution actually look like, and does it decompose?
2. Does the mean vary by anything observable (airport, month, stand, proxy), or is one
   constant as good as it gets?
3. Is the shipped delta model beating the alternatives, or just beating the old linear one?

Everything is scored on the regime rows alone (plain MSE on that subpopulation), which is the
quantity the weighted metric is a monotone function of once the other rows are held fixed.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import model_features
from src.postprocess import apply_regime_model, fit_regime_model
from src.validation import mirrored_split

d = pd.read_parquet("data/processed/train_features.parquet")
d = d.merge(pd.read_parquet("data/processed/train_surface.parquet"),
            on="MVT_ID_mvt", how="left", validate="one_to_one")
is_train, is_val, cell = mirrored_split(d)
R = d[d.dq_no_aircraft_type & d.TAXITIME_SEC_mvt.notna()].copy()
R["month"] = R.MVT_TIME_UTC_mvt.dt.month
y = R.TAXITIME_SEC_mvt

print(f"=== {len(R):,} regime rows across the labelled year ===")
print(f"mean {y.mean():,.0f}s  median {y.median():,.0f}s  sd {y.std():,.0f}s  max {y.max():,.0f}s")
print("\nquantiles:")
print(y.quantile([.05,.1,.25,.5,.75,.9,.95,.99]).round(0).to_string())

print("\n=== does the target decompose? distance to the nearest whole hour / day ===")
for unit, n in [("hour", 3600), ("day", 86400)]:
    r = (y % n)
    near = np.minimum(r, n - r)
    print(f"  {unit:<5}: median distance {near.median():>8,.0f}s   "
          f"within 300s of a whole {unit}: {(near < 300).mean():>6.1%}   "
          f"(uniform would give {2*300/n:.1%})")

print("\n=== is the population a mixture of 'ordinary taxi' and 'broken clock'? ===")
ordinary = y < 3600
print(f"  under 1h: {ordinary.sum():>5,} rows ({ordinary.mean():.1%}), mean {y[ordinary].mean():>8,.0f}s")
print(f"  over  1h: {(~ordinary).sum():>5,} rows ({(~ordinary).mean():.1%}), mean {y[~ordinary].mean():>8,.0f}s")
print(f"  a predictor that knew which: MSE floor "
      f"{np.sqrt((ordinary*(y-y[ordinary].mean())**2 + (~ordinary)*(y-y[~ordinary].mean())**2).mean()):,.0f}s"
      f"  vs one constant {y.std():,.0f}s")

print("\n=== the quantity the delta model actually predicts: BLOCK - SCHED ===")
print("taxi = MVT - BLOCK and proxy_taxi_sched = MVT - SCHED, so the residual the delta\n"
      "model regresses is exactly the off-block delay against schedule. If regime rows had\n"
      "an ordinary departure delay, this branch would already be near its floor.\n")
d["_bs_tmp"] = (d.BLOCK_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds()
reg = d.dq_no_aircraft_type & d.TAXITIME_SEC_mvt.notna()
nrm = ~d.dq_no_aircraft_type & d.TAXITIME_SEC_mvt.notna()
for nm, m in [("regime rows", reg), ("normal rows", nrm)]:
    q = d.loc[m, "_bs_tmp"].quantile([.05,.25,.5,.75,.95])
    print(f"  {nm:<12} n={int(m.sum()):>9,}  mean {d.loc[m,"_bs_tmp"].mean():>9,.0f}s  "
          f"sd {d.loc[m,"_bs_tmp"].std():>9,.0f}s  "
          f"p5 {q[.05]:>8,.0f}  p50 {q[.5]:>7,.0f}  p95 {q[.95]:>9,.0f}")
print("\n  -> the delta model's irreducible floor on regime rows is the sd of this column,\n"
      "     conditional on whatever it can observe.")

print("\n=== can a regime row be matched to the same flight number on other days? ===")
print("Regime rows have no flight-plan record at all, but FLIGHT_mvt, STAND_mvt and\n"
      "RUNWAY_mvt survive. If the same flight number flies the same route on other days with\n"
      "an intact record, its usual behaviour is observable.\n")
d["_key"] = d.ADEP_mvt.astype(str) + "|" + d.FLIGHT_mvt.astype(str)
donor = d[nrm & d.FLIGHT_mvt.notna()]
dg = donor.groupby("_key", observed=True)["TAXITIME_SEC_mvt"].agg(["size", "mean", "median"])
dg = dg[dg["size"] >= 3]
rk = d.loc[reg, "_key"]
hit = rk.isin(dg.index)
print(f"  regime rows with >=3 same-flight-number donors elsewhere: {hit.mean():.1%} "
      f"({int(hit.sum()):,} of {int(reg.sum()):,})")
if hit.any():
    j = d.loc[reg][hit.to_numpy()].join(dg[["mean", "median"]], on="_key")
    e = j["mean"] - j.TAXITIME_SEC_mvt
    print(f"  using the donor mean as the prediction: RMSE {np.sqrt((e**2).mean()):,.0f}s  "
          f"vs constant {np.sqrt(((j.TAXITIME_SEC_mvt - y.mean())**2).mean()):,.0f}s")
    print(f"  (donor mean is a *normal* taxi, ~{j['mean'].median():,.0f}s median, against a "
          f"regime target median of {j.TAXITIME_SEC_mvt.median():,.0f}s)")

print("\n=== does the mean vary by anything observable? ===")
for k in ["ADEP_mvt", "month", "stand_area"]:
    g = R.groupby(k, observed=True)["TAXITIME_SEC_mvt"].agg(["size", "mean", "median"])
    g = g[g["size"] >= 20].sort_values("mean", ascending=False)
    print(f"\n  by {k} (groups with >=20 rows):")
    print(g.head(10).round(0).to_string())

print("\n=== out-of-sample comparison of predictors, leave-one-month-out ===")
print("Each month is held out, every predictor fitted on the other eleven, scored on the\n"
      "held-out month's regime rows. RMSE in seconds; lower is better.\n")
nums, cats = model_features(d.drop(columns=["_bs_tmp", "_key"], errors="ignore"))
for c in cats:
    d[c] = d[c].astype("string").fillna("__NA__").astype("category")
    if c in R.columns:
        R[c] = R[c].astype("string").fillna("__NA__").astype("category")
feats = nums + cats

rows = []
for mth in range(1, 13):
    te = R[R.month == mth]
    trm = R[R.month != mth]
    if len(te) < 10:
        continue
    yt = te.TAXITIME_SEC_mvt.to_numpy()
    cand = {
        "constant mean": np.full(len(te), trm.TAXITIME_SEC_mvt.mean()),
        "constant median": np.full(len(te), trm.TAXITIME_SEC_mvt.median()),
        "per-airport mean": te.ADEP_mvt.map(
            trm.groupby("ADEP_mvt", observed=True)["TAXITIME_SEC_mvt"].mean()
        ).fillna(trm.TAXITIME_SEC_mvt.mean()).to_numpy(),
        "proxy_taxi_sched": te.proxy_taxi_sched.fillna(trm.TAXITIME_SEC_mvt.mean()).to_numpy(),
    }
    # the shipped delta model, refitted without the held-out month
    src = d[d.MVT_TIME_UTC_mvt.dt.month != mth]
    try:
        model = fit_regime_model(src, feats)
        base = np.full(len(te), np.nan)
        pr, _ = apply_regime_model(te, base, model)
        cand["delta regime model (shipped)"] = pr
    except Exception as exc:                                    # noqa: BLE001
        cand["delta regime model (shipped)"] = np.full(len(te), np.nan)
        print(f"  (month {mth}: regime model failed -- {exc})")
    # a winsorised mean: the sample mean of 1.4k heavy-tailed rows is itself noisy
    for cap in (7200, 14400, 28800):
        w = trm.TAXITIME_SEC_mvt.clip(upper=cap)
        cand[f"mean winsorised at {cap//3600}h"] = np.full(len(te), w.mean())
    rows.append({"month": mth, "n": len(te),
                 **{k: np.sqrt(np.nanmean((v - yt) ** 2)) for k, v in cand.items()}})

t = pd.DataFrame(rows).set_index("month")
n = t.pop("n")
print(t.round(0).to_string())
print("\nweighted by row count (the number that matters):")
w = n / n.sum()
print((t.pow(2).mul(w, axis=0).sum() ** 0.5).sort_values().round(1).to_string())
print(f"\n(row counts per month: {n.to_dict()})")
