#!/usr/bin/env python
"""E22: the model under-predicts long taxis. Is that recoverable, or is it an oracle?

E21 found mean residuals of -322s, -871s and -1856s in the 1800-2700, 2700-3600 and 3600+
true-taxi bands. That is conditioning on the truth, so it proves nothing actionable on its
own -- regression to the mean produces exactly that pattern even in a perfectly calibrated
model, because the bands are selected on y.

The question that *is* actionable: conditioned on what the model **predicts**, is the mean
outcome higher than the prediction? If E[y | pred] > pred at the top of the range, the model
is under-dispersed in a way a monotone recalibration can fix, and the fix is available at
prediction time. If E[y | pred] == pred, the bands in E21 are pure selection and there is
nothing to recover without better features.

Also checks whether any such miscalibration is stable enough across months to transfer, since
a correction fitted on Jan-Jun and applied to Dec/Jul is the only honest way to use it.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.validation import mirrored_split, ranking_composition, weighted_rmse

d = pd.read_parquet("data/processed/train_features.parquet")
is_train, is_val, cell = mirrored_split(d)
val = d[is_val].copy()
pred = np.load("data/processed/exp_cache/r4_pred.npy")
val["pred"], val["cell"] = pred, cell[is_val].to_numpy()
y = val["TAXITIME_SEC_mvt"].to_numpy()
normal = ~(val.dq_no_aircraft_type | val.dq_implausible_target).to_numpy()
N = val[normal].copy()

print("=== calibration: bin by PREDICTED value (available at prediction time) ===")
edges = [0, 600, 800, 1000, 1200, 1400, 1600, 1800, 2100, 2400, 3000, 1e9]
N["bin"] = pd.cut(N.pred, edges)
g = N.groupby("bin", observed=True)
tbl = pd.DataFrame({
    "n": g.size(),
    "mean_pred": g["pred"].mean(),
    "mean_actual": g["TAXITIME_SEC_mvt"].mean(),
    "gap": g["TAXITIME_SEC_mvt"].mean() - g["pred"].mean(),
    "rmse": np.sqrt(g.apply(lambda x: ((x.pred - x.TAXITIME_SEC_mvt) ** 2).mean(),
                            include_groups=False)),
})
print(tbl.round(1).to_string())
print(f"\noverall mean gap (actual - pred), normal rows: {(N.TAXITIME_SEC_mvt - N.pred).mean():+.1f}s")

print("\n=== is the gap stable across the two validation cells? ===")
for c in ["jan", "jul"]:
    s = N[N.cell == c]
    gg = s.groupby(pd.cut(s.pred, edges), observed=True).apply(
        lambda x: pd.Series({"n": len(x), "gap": (x.TAXITIME_SEC_mvt - x.pred).mean()}),
        include_groups=False)
    print(f"  {c}: " + "  ".join(f"{int(r.n):>6}:{r.gap:+7.0f}" for _, r in gg.iterrows()))

print("\n=== what an isotonic recalibration fitted OUT OF SAMPLE would buy ===")
# fit the mapping on the training window's own out-of-fold-ish predictions is not available
# here, so use the honest proxy: fit on `jan` (Dec-2025) and apply to `jul`, and vice versa.
from sklearn.isotonic import IsotonicRegression
comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
vcell, vapt = cell[is_val], val["ADEP_mvt"]
base = weighted_rmse(y, pred, vcell, vapt, comp)
base_t = weighted_rmse(y[normal], pred[normal], vcell[normal], vapt[normal], comp)
print(f"  baseline (no post-processing here)   full {base:7.2f}  normal-only {base_t:7.2f}")

adj = pred.copy()
ncell = N.cell.to_numpy()
for fit_on, apply_to in [("jan", "jul"), ("jul", "jan")]:
    f, a = ncell == fit_on, ncell == apply_to
    iso = IsotonicRegression(out_of_bounds="clip").fit(N.pred[f], N.TAXITIME_SEC_mvt[f])
    idx = np.flatnonzero(normal)[a]
    adj[idx] = iso.predict(pred[idx])
print(f"  + isotonic, fitted on the other cell  full "
      f"{weighted_rmse(y, adj, vcell, vapt, comp):7.2f}  normal-only "
      f"{weighted_rmse(y[normal], adj[normal], vcell[normal], vapt[normal], comp):7.2f}")

print("\n=== where DO the long true taxis come from? proxy behaviour in the tail ===")
print("If the proxy tracked the target, a long taxi would already be predicted long.")
cols = ["TAXITIME_SEC_mvt", "proxy_taxi_aobt3", "proxy_taxi_eobt1", "proxy_taxi_sched",
        "offblock_delay_sec", "pred"]
for lo, hi in [(0, 1200), (1200, 1800), (1800, 2700), (2700, 3600), (3600, 1e9)]:
    m = (N.TAXITIME_SEC_mvt >= lo) & (N.TAXITIME_SEC_mvt < hi)
    s = N.loc[m, cols].median()
    print(f"  taxi [{lo:>5},{hi:>7.0f})  n={int(m.sum()):>7,}  " +
          "  ".join(f"{c.replace('proxy_taxi_','px_').replace('TAXITIME_SEC_mvt','y'):<14}"
                    f"{s[c]:>7.0f}" for c in cols))

print("\n=== the unobservable part: AOBT_3 - BLOCK_TIME, by true-taxi band (train rows) ===")
t = d[is_train].copy()
t = t[trainable_mask(t) & t.AOBT_3_flt.notna() & t.BLOCK_TIME_UTC_mvt.notna()]
t["hold"] = (t.AOBT_3_flt - t.BLOCK_TIME_UTC_mvt).dt.total_seconds()
for lo, hi in [(0, 1200), (1200, 1800), (1800, 2700), (2700, 3600), (3600, 1e9)]:
    m = (t.TAXITIME_SEC_mvt >= lo) & (t.TAXITIME_SEC_mvt < hi)
    q = t.hold[m].quantile([0.1, 0.5, 0.9])
    print(f"  taxi [{lo:>5},{hi:>7.0f})  n={int(m.sum()):>8,}  "
          f"hold p10 {q[0.1]:>7.0f}  p50 {q[0.5]:>7.0f}  p90 {q[0.9]:>7.0f}  "
          f"mean {t.hold[m].mean():>7.0f}")
