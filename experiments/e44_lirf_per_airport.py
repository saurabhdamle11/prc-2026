#!/usr/bin/env python
"""E44: the (airport x month) matrix under v5, LIRF forensics, per-airport models, a LIRF-only cap.

`e19`/`e21` established that `jul LIRF` dominates the metric and is seasonal, not a 2025
artifact — but against the round-4 model. This re-measures the matrix against v5 (using e43's
cached single-seed `ref` predictions, no refit) and splits each cell into normal vs
missing-aircraft-type rows, then tests the two ideas nobody measured:

* **per-airport models** — ten LightGBM fits, each on one airport's rows, same v5 parameters.
  A global model shares splits across very different layouts; the question is whether
  airport-specific interactions are left on the table. Reported both as "all ten swapped in"
  and "LIRF swapped in only" (the latter is chosen a priori, not by looking at the table).
* **LIRF-only target cap** — the global p99.5/p99.9 caps lost 10.9/2.9 s (`e43`). A cap scoped
  to the one airport with the suspect tail is a different hypothesis.

Forensics (no model): LIRF monthly taxi distribution and runway mix across 2025, and the
`AOBT_3_flt - BLOCK_TIME_UTC_mvt` gap by airport — if LIRF records off-block at a different
milestone, the target itself is noisier there.

Arms: xtab, forensics, per_airport, lirfcap995, lirfcap999. Appends to exp_cache/e44.jsonl.
"""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.config_features import CONFIG_CATEGORICALS
from src.features import model_features
from src.offblock_gaps import add_offblock_gaps
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)
from src.validation import mirrored_split, ranking_composition, report

OFFSET = 5.73
CONFIG_PREFIXES = ("depcfg_", "arrcfg_", "stand_turnaround", "stand_prev_arr")
OUT = Path("data/processed/exp_cache")
ARMS = sys.argv[1:] or ["xtab", "forensics", "per_airport", "lirfcap995", "lirfcap999"]

raw = json.loads(Path("data/processed/lgbm_best_params_r3.json").read_text())
INT_P = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}
p = {k: (int(v) if k in INT_P else v) for k, v in raw.items() if not k.startswith("_")}
n_est = p.pop("n_estimators")
SMOKE = "E44_NEST" in os.environ
n_est = int(os.environ.get("E44_NEST", n_est))
PARAMS = dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1, subsample_freq=1,
              n_estimators=n_est, cat_smooth=100.0, **p)
LOG = OUT / ("e44_smoke.jsonl" if SMOKE else "e44.jsonl")

comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
for extra in ("train_surface", "train_config", "train_osm"):
    d = d.merge(pd.read_parquet(f"data/processed/{extra}.parquet"),
                on="MVT_ID_mvt", how="left", validate="one_to_one")
d, gaps = add_offblock_gaps(d)
nums, cats = model_features(d)
cats = list(cats) + [c for c in CONFIG_CATEGORICALS if c in d.columns]
for c in cats:
    d[c] = d[c].astype("string").fillna("__NA__").astype("category")
feats = nums + cats

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


def score(pred_raw, label, t0, extra=None):
    np.save(OUT / f"e44_pred_{label}.npy", pred_raw)
    r = report(yv, post(pred_raw), val, comp, vcell, trimmed_mask=trimmed)
    row = {"arm": label, "full": r["rmse_full"], "trimmed": r["rmse_trimmed"],
           "board_est": r["rmse_full"] + OFFSET, "sec": round(time.time() - t0), **(extra or {})}
    print(f"{label:<14} full {row['full']:>7.2f}  trimmed {row['trimmed']:>7.2f}  "
          f"board~ {row['board_est']:>7.2f}   ({row['sec']}s)", flush=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def row_weights():
    """Per-row weights such that sum(w * sq) is the leaderboard-weighted MSE."""
    f = pd.DataFrame({"slice": vcell.to_numpy(), "airport": apt_v})
    f = f.merge(comp[["slice", "airport", "weight"]], on=["slice", "airport"], how="left")
    n = f.groupby(["slice", "airport"])["weight"].transform("size")
    w = (f["weight"] / n).fillna(0).to_numpy()
    return w / w.sum()


def per_airport_table(pred, label):
    w, sq = row_weights(), (pred - yv) ** 2
    f = pd.DataFrame({"airport": apt_v, "wsq": w * sq, "w": w})
    g = f.groupby("airport").agg(w=("w", "sum"), wsq=("wsq", "sum"))
    return pd.Series(np.sqrt(g["wsq"] / g["w"]), name=label)


def fit(X_tr, target, X_val):
    m = lgb.LGBMRegressor(**PARAMS, random_state=0).fit(X_tr, target)
    pv = m.predict(X_val)
    del m; gc.collect()
    return pv


for arm in ARMS:
    t0 = time.time()
    print(f"\n=== {arm} ===", flush=True)

    if arm == "xtab":
        pred = post(np.load(OUT / "e43_pred_ref.npy"))
        w, sq = row_weights(), (pred - yv) ** 2
        noact = val["dq_no_aircraft_type"].to_numpy()
        f = pd.DataFrame({"airport": apt_v, "slice": vcell.to_numpy(), "w": w, "wsq": w * sq,
                          "wsq_noact": w * sq * noact, "noact": noact})
        tot = f["wsq"].sum()
        g = f.groupby(["airport", "slice"]).agg(
            n=("w", "size"), n_noact=("noact", "sum"), w=("w", "sum"),
            wsq=("wsq", "sum"), wsq_noact=("wsq_noact", "sum"))
        norm_w = f[~f["noact"]].groupby(["airport", "slice"])["w"].sum()
        t = pd.DataFrame({
            "n": g["n"], "n_noact": g["n_noact"], "lb_weight_%": 100 * g["w"],
            "rmse": np.sqrt(g["wsq"] / g["w"]),
            "rmse_normal": np.sqrt((g["wsq"] - g["wsq_noact"]) / norm_w),
            "mse_share_%": 100 * g["wsq"] / tot,
            "share_from_noact_%": 100 * g["wsq_noact"] / tot})
        print(t.round(1).to_string(), flush=True)
        print("\nrmse_normal pivot (airport x slice):")
        print(t["rmse_normal"].unstack().round(1).to_string(), flush=True)
        t.round(3).reset_index().to_csv(OUT / "e44_xtab.csv", index=False)

    elif arm == "forensics":
        tr_ok = (trainable & d["TAXITIME_SEC_mvt"].notna()).to_numpy()
        month = d["MVT_TIME_UTC_mvt"].dt.month.to_numpy()
        L = pd.DataFrame({"m": month, "y": y, "rwy": d["RUNWAY_mvt"].astype("string"),
                          "noact": d["dq_no_aircraft_type"].to_numpy()})[(apt_all == "LIRF")]
        Lt = L[tr_ok[apt_all == "LIRF"]]
        q = Lt.groupby("m")["y"].describe(percentiles=[.5, .9, .99])[["count", "mean", "50%", "90%", "99%"]]
        q["share_>1800s_%"] = Lt.groupby("m")["y"].apply(lambda s: 100 * (s > 1800).mean())
        q["noact_rows"] = L.groupby("m")["noact"].sum()
        print("LIRF taxi-time by month (trainable rows; noact over all rows):")
        print(q.round(1).to_string())
        mix = pd.crosstab(Lt["m"], Lt["rwy"], normalize="index").mul(100)
        mix = mix.loc[:, mix.max() > 2]
        print("\nLIRF runway mix by month, % of departures (runways > 2% in any month):")
        print(mix.round(1).to_string())
        gap = (d["AOBT_3_flt"] - d["BLOCK_TIME_UTC_mvt"]).dt.total_seconds()
        G = pd.DataFrame({"apt": apt_all, "gap": gap})[tr_ok & is_train.to_numpy()].dropna()
        gt = G.groupby("apt")["gap"].agg(
            median="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75),
            abs_gt_600s_pct=lambda s: 100 * (s.abs() > 600).mean())
        gt["iqr"] = gt["q75"] - gt["q25"]
        print("\nAOBT_3_flt - BLOCK_TIME_UTC_mvt by airport, seconds (train months, trainable):")
        print(gt.round(1).to_string(), flush=True)
        q.round(2).reset_index().to_csv(OUT / "e44_lirf_months.csv", index=False)
        mix.round(2).reset_index().to_csv(OUT / "e44_lirf_runways.csv", index=False)
        gt.round(2).reset_index().to_csv(OUT / "e44_aobt3_gap.csv", index=False)

    elif arm == "per_airport":
        ref = np.load(OUT / "e43_pred_ref.npy")
        pa = ref.copy()
        Xtr_all, Xv_all = d.iloc[tr_idx][feats], val[feats]
        for a in sorted(np.unique(apt_v)):
            mt, mv = apt_tr == a, apt_v == a
            ta = time.time()
            pa[mv] = fit(Xtr_all[mt], ytr[mt], Xv_all[mv])
            print(f"  {a}: fit {mt.sum():,} rows, {time.time() - ta:.0f}s", flush=True)
        score(pa, "per_airport_all", t0)
        lirf_only = ref.copy()
        lirf_only[apt_v == "LIRF"] = pa[apt_v == "LIRF"]
        score(lirf_only, "per_airport_lirf", t0)
        cmp = pd.concat([per_airport_table(post(ref), "global"),
                         per_airport_table(post(pa), "per_airport")], axis=1)
        cmp["delta"] = cmp["per_airport"] - cmp["global"]
        print("\nper-airport weighted RMSE, global vs own model (descriptive only — "
              "picking winners from this table would be selection on the holdout):")
        print(cmp.round(1).to_string(), flush=True)
        cmp.round(3).reset_index().to_csv(OUT / "e44_per_airport.csv", index=False)

    elif arm.startswith("lirfcap"):
        q = float("0." + arm.removeprefix("lirfcap"))
        is_l = apt_tr == "LIRF"
        cap = float(np.quantile(ytr[is_l], q))
        target = np.where(is_l, np.minimum(ytr, cap), ytr)
        print(f"  LIRF cap at p{q*100:g} = {cap:.0f}s, clips {(is_l & (ytr > cap)).sum():,} rows",
              flush=True)
        score(fit(d.iloc[tr_idx][feats], target, val[feats]), arm, t0, {"q": q, "cap_s": cap})

    else:
        raise SystemExit(f"unknown arm {arm}")
