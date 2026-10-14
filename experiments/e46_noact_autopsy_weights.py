#!/usr/bin/env python
"""E46: autopsy of the missing-aircraft-type rows, and label-noise-aware sample weights.

* **autopsy** (no fit): what the 428 validation missing-type rows (337 of them `jul LIRF`) are —
  flight-record join status, placeholder labels, target distribution — and how the shipped
  regime branch compares with dumb baselines on exactly those rows, including the total-score
  effect of substituting each baseline.
* **sample weights**: down-weight training rows whose ADS-B first movement contradicts the
  reported off-block. The gap is centred on each airport's training median first, because
  LTFM's median is -301 s (a systematic offset, not noise). Two arms, both 0.25 weight:
    - `w_abs_rt`: |centred gap| > 600 s at LIRF and LTFM only (the proposal as stated);
    - `w_neg`: centred gap < -600 s anywhere — ADS-B moving >10 min *before* the reported
      off-block, which is physically contradictory. Positive gaps are left alone because
      `e22` showed they are mostly the real post-pushback hold.
  Weights touch training only; inference is unchanged, so there is no leakage.
One seed each on the Oracle VM, against e43's `ref` (274.40 / 230.92).
"""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.config_features import CONFIG_CATEGORICALS
from src.features import model_features
from src.metar_features import METAR_FEATURES
from src.offblock_gaps import add_offblock_gaps
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)
from src.validation import mirrored_split, ranking_composition, report

OFFSET = 5.73
CONFIG_PREFIXES = ("depcfg_", "arrcfg_", "stand_turnaround", "stand_prev_arr")
OUT = Path("data/processed/exp_cache")
ARMS = sys.argv[1:] or ["autopsy", "w_abs_rt", "w_neg"]

raw = json.loads(Path("data/processed/lgbm_best_params_r3.json").read_text())
INT_P = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}
p = {k: (int(v) if k in INT_P else v) for k, v in raw.items() if not k.startswith("_")}
n_est = p.pop("n_estimators")
SMOKE = "E46_NEST" in os.environ
n_est = int(os.environ.get("E46_NEST", n_est))
PARAMS = dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1, subsample_freq=1,
              n_estimators=n_est, cat_smooth=100.0, **p)
LOG = OUT / ("e46_smoke.jsonl" if SMOKE else "e46.jsonl")

comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
for extra in ("train_surface", "train_config", "train_osm", "train_metar"):
    d = d.merge(pd.read_parquet(f"data/processed/{extra}.parquet"),
                on="MVT_ID_mvt", how="left", validate="one_to_one")
d, gaps = add_offblock_gaps(d)
nums, cats = model_features(d)
cats = list(cats) + [c for c in CONFIG_CATEGORICALS if c in d.columns]
for c in cats:
    d[c] = d[c].astype("string").fillna("__NA__").astype("category")
feats = [c for c in nums + cats if c not in METAR_FEATURES]  # v5 exactly

is_train, is_val, cell = mirrored_split(d)
trainable = trainable_mask(d)
fit_mask = (is_train & trainable).to_numpy()
tr_idx, val_idx = np.flatnonzero(fit_mask), np.flatnonzero(is_val.to_numpy())
regime = fit_regime_model(d[is_train], [c for c in feats if not c.startswith(CONFIG_PREFIXES)])
val = d.iloc[val_idx].copy()
vcell, trimmed = cell.iloc[val_idx], trainable_mask(val)
y = d["TAXITIME_SEC_mvt"].to_numpy("float64")
ytr, yv = y[tr_idx], y[val_idx]
apt_all = d["ADEP_mvt"].astype("string").to_numpy()
apt_tr, apt_v = apt_all[tr_idx], apt_all[val_idx]
print(f"v5 {len(feats)} feats | fit {len(tr_idx):,} | val {len(val_idx):,} | arms {ARMS}\n",
      flush=True)




def post(pred_raw):
    pred, _ = apply_regime_model(val, pred_raw, regime)
    pred, _ = apply_extreme_proxy_override(val, pred)
    return pred


def row_weights():
    f = pd.DataFrame({"slice": vcell.to_numpy(), "airport": apt_v})
    f = f.merge(comp[["slice", "airport", "weight"]], on=["slice", "airport"], how="left")
    n = f.groupby(["slice", "airport"])["weight"].transform("size")
    w = (f["weight"] / n).fillna(0).to_numpy()
    return w / w.sum()



def fit_score(weights, label):
    t0 = time.time()
    m = lgb.LGBMRegressor(**PARAMS, random_state=0).fit(d.iloc[tr_idx][feats], ytr,
                                                        sample_weight=weights)
    pv = m.predict(val[feats]); del m; gc.collect()
    np.save(OUT / f"e46_pred_{label}.npy", pv)
    r = report(yv, post(pv), val, comp, vcell, trimmed_mask=trimmed)
    row = {"arm": label, "full": r["rmse_full"], "trimmed": r["rmse_trimmed"],
           "board_est": r["rmse_full"] + OFFSET, "sec": round(time.time() - t0),
           "downweighted": int((weights < 1).sum())}
    print(f"{label:<10} full {row['full']:>7.2f}  trimmed {row['trimmed']:>7.2f}  "
          f"board~ {row['board_est']:>7.2f}  ({row['downweighted']:,} rows at 0.25, {row['sec']}s)"
          f"   [ref 274.40 / 230.92]", flush=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


gap = (d["AOBT_3_flt"] - d["BLOCK_TIME_UTC_mvt"]).dt.total_seconds().to_numpy()
gap_tr = gap[tr_idx]
med = pd.Series(gap_tr).groupby(apt_tr).median()
cgap = gap_tr - pd.Series(apt_tr).map(med).to_numpy()

for arm in ARMS:
    print(f"\n=== {arm} ===", flush=True)
    if arm == "autopsy":
        raw_ref = np.load(OUT / "e43_pred_ref.npy")
        shipped = post(raw_ref)
        na = val["dq_no_aircraft_type"].to_numpy()
        jl = na & (apt_v == "LIRF") & (vcell.to_numpy() == "jul")
        V = val[na]
        flt = [c for c in val.columns if c.endswith("_flt")]
        print(f"missing-type validation rows: {na.sum()} (jul LIRF {jl.sum()})")
        print(f"  dq_no_flight_record: {V['dq_no_flight_record'].mean()*100:.1f}%   "
              f"all {len(flt)} _flt columns null: {V[flt].isna().all(axis=1).mean()*100:.1f}%   "
              f"AOBT_3 present: {V['AOBT_3_flt'].notna().mean()*100:.1f}%")
        ph = (V["BLOCK_TIME_UTC_mvt"] == V["SCHED_TIME_UTC_mvt"]).to_numpy()
        print(f"  placeholder BLOCK == SCHED: {ph.mean()*100:.1f}%   STAND present: "
              f"{V['STAND_mvt'].notna().mean()*100:.1f}%   RUNWAY present: {V['RUNWAY_mvt'].notna().mean()*100:.1f}%")
        q = [.1, .25, .5, .75, .9, .99]
        yy = yv[na]
        print("  target quantiles (s): " + "  ".join(f"p{int(x*100)} {np.quantile(yy, x):.0f}" for x in q))
        print(f"  target by label type: placeholder median {np.median(yy[ph]):.0f}s, "
              f"non-placeholder median {np.median(yy[~ph]):.0f}s")
        # baselines, each substituted on the missing-type rows only
        trn = d.iloc[tr_idx]
        hr = d["local_hour"].to_numpy()
        ah = pd.Series(ytr).groupby([apt_tr, hr[tr_idx]]).median()
        ah_pred = pd.Series(list(zip(apt_v, hr[val_idx]))).map(ah).to_numpy(dtype="float64")
        tr_na = d[is_train & d["dq_no_aircraft_type"]]
        cands = {
            "shipped regime branch": shipped,
            "main model (no branch)": raw_ref,
            "proxy_taxi_sched": val["proxy_taxi_sched"].fillna(regime["constant_fallback"]).to_numpy(),
            "airport-hour median (normal)": ah_pred,
            "missing-type train mean": np.full(len(yv), tr_na["TAXITIME_SEC_mvt"].mean()),
        }
        w = row_weights()
        print(f"\n  {'predictor':<30} {'rmse (all 428)':>15} {'rmse (jul LIRF)':>16} {'total full':>11}")
        for k, pk in cands.items():
            sub = shipped.copy(); sub[na] = pk[na]
            tot = np.sqrt((w * (sub - yv) ** 2).sum())
            r_all = np.sqrt(np.mean((pk[na] - yv[na]) ** 2))
            r_jl = np.sqrt(np.mean((pk[jl] - yv[jl]) ** 2))
            print(f"  {k:<30} {r_all:>15.0f} {r_jl:>16.0f} {tot:>11.2f}", flush=True)
        print(f"  (total if these rows were predicted perfectly: "
              f"{np.sqrt((w * np.where(na, 0, (shipped - yv) ** 2)).sum()):.2f})")
    elif arm == "w_abs_rt":
        wt = np.where(np.isin(apt_tr, ["LIRF", "LTFM"]) & (np.abs(cgap) > 600), 0.25, 1.0)
        fit_score(wt, arm)
    elif arm == "w_neg":
        wt = np.where(cgap < -600, 0.25, 1.0)
        fit_score(wt, arm)
    else:
        raise SystemExit(f"unknown arm {arm}")
