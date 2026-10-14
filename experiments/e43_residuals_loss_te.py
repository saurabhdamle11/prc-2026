#!/usr/bin/env python
"""E43: residual breakdown of v5, then loss/target transforms and out-of-fold target encoding.

Three questions none of rounds 2-4 measured:

1. **Where is v5 wrong?** Share of the *leaderboard-weighted* MSE by true taxi-time bucket
   (<10 / 10-20 / 20-30 / 30-60 / 60+ min), by airport and by local hour, with the signed bias
   in each. `e19` budgeted by (slice, airport) cell and regime only.
2. **Does the loss or the target scale help the bulk?** Huber at two deltas, the target
   winsorised at two quantiles, and log1p with Duan smearing on the way back. Recalibration of
   the *predictions* is closed (`e22`); these change what the trees fit, which is different.
3. **Does out-of-fold target encoding add anything over native categoricals + cat_smooth?**
   Month-grouped OOF means (smoothed toward the airport mean) for (airport, hour),
   (airport, runway), (airport, stand), aircraft type, (airport, aircraft type) and
   (airport, stand, runway). Validation rows get the encoding fitted on all training months.

Every arm is one seed of the exact v5 configuration (`e38`) with the post-processing rules
applied unchanged, so each is compared against arm `ref` on this machine only — see the
thread-count caveat in SKILLS.md §11. Results append to data/processed/exp_cache/e43.jsonl.
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
from src.osm_features import OSM_FEATURES  # noqa: F401  (imported for parity with e38)
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)
from src.validation import mirrored_split, ranking_composition, report

OFFSET = 5.73
CONFIG_PREFIXES = ("depcfg_", "arrcfg_", "stand_turnaround", "stand_prev_arr")
OUT = Path("data/processed/exp_cache")
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "e43.jsonl"
ARMS = sys.argv[1:] or ["ref", "te", "huber600", "huber1200", "win995", "win999", "log"]

raw = json.loads(Path("data/processed/lgbm_best_params_r3.json").read_text())
INT_P = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}
p = {k: (int(v) if k in INT_P else v) for k, v in raw.items() if not k.startswith("_")}
n_est = p.pop("n_estimators")
SMOKE = "E43_NEST" in os.environ  # e.g. E43_NEST=50 for a quick end-to-end check
n_est = int(os.environ.get("E43_NEST", n_est))
PARAMS = dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1, subsample_freq=1,
              n_estimators=n_est, cat_smooth=100.0, **p)

comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
for extra in ("train_surface", "train_config", "train_osm"):
    d = d.merge(pd.read_parquet(f"data/processed/{extra}.parquet"),
                on="MVT_ID_mvt", how="left", validate="one_to_one")
d, gaps = add_offblock_gaps(d)
nums, cats = model_features(d)
cats = list(cats) + [c for c in CONFIG_CATEGORICALS if c in d.columns]

# --- target-encoding keys, built from raw strings before categorical encoding ---------------
def _s(c):
    return d[c].astype("string").fillna("__NA__")

TE_KEYS = {
    "te_apt_hour": _s("ADEP_mvt") + "|" + d["local_hour"].astype("string"),
    "te_apt_rwy": _s("ADEP_mvt") + "|" + _s("RUNWAY_mvt"),
    "te_apt_stand": _s("ADEP_mvt") + "|" + _s("stand_full"),
    "te_actype": _s("AIRCRAFT_TYPE_mvt"),
    "te_apt_actype": _s("ADEP_mvt") + "|" + _s("AIRCRAFT_TYPE_mvt"),
    "te_apt_stand_rwy": _s("ADEP_mvt") + "|" + _s("stand_full") + "|" + _s("RUNWAY_mvt"),
}

for c in cats:
    d[c] = d[c].astype("string").fillna("__NA__").astype("category")
v5_feats = nums + cats

is_train, is_val, cell = mirrored_split(d)
trainable = trainable_mask(d)
fit_mask = (is_train & trainable).to_numpy()
tr_idx, val_idx = np.flatnonzero(fit_mask), np.flatnonzero(is_val.to_numpy())
regime = fit_regime_model(d[is_train], [c for c in v5_feats if not c.startswith(CONFIG_PREFIXES)])
val = d.iloc[val_idx].copy()
vcell, trimmed = cell.iloc[val_idx], trainable_mask(val)
y = d["TAXITIME_SEC_mvt"].to_numpy("float64")
ytr, yv = y[tr_idx], y[val_idx]
print(f"v5 {len(v5_feats)} feats | fit {len(tr_idx):,} | val {len(val_idx):,} | arms {ARMS}\n",
      flush=True)


def smoothed_mean(key_fit, y_fit, prior_fit, key_apply, prior_apply, m=100.0):
    g = pd.DataFrame({"k": key_fit, "y": y_fit, "p": prior_fit}).groupby("k", observed=True)
    stats = pd.DataFrame({"s": g["y"].sum(), "n": g["y"].size()})
    j = pd.DataFrame({"k": key_apply}).join(stats, on="k")
    s, n = j["s"].fillna(0).to_numpy(), j["n"].fillna(0).to_numpy()
    return (s + m * prior_apply) / (n + m)


def build_te():
    """Month-grouped OOF encodings on the fit rows; full-train encodings on validation."""
    apt = d["ADEP_mvt"].astype("string").to_numpy()
    month = d["MVT_TIME_UTC_mvt"].dt.month.to_numpy()
    apt_mean = pd.Series(ytr).groupby(apt[tr_idx]).mean()
    prior = pd.Series(apt).map(apt_mean).fillna(ytr.mean()).to_numpy()
    cols = {}
    for name, key in TE_KEYS.items():
        k = key.to_numpy()
        col = np.full(len(d), np.nan)
        for mth in np.unique(month[tr_idx]):
            ins = tr_idx[month[tr_idx] != mth]
            out = tr_idx[month[tr_idx] == mth]
            col[out] = smoothed_mean(k[ins], y[ins], prior[ins], k[out], prior[out])
        col[val_idx] = smoothed_mean(k[tr_idx], ytr, prior[tr_idx], k[val_idx], prior[val_idx])
        cols[name] = col
    return pd.DataFrame(cols, index=d.index)


def breakdown(pred, label):
    """Share of leaderboard-weighted MSE, and signed bias, by bucket / airport / local hour."""
    w_cell = comp.set_index(["slice", "airport"])["weight"]
    key = pd.MultiIndex.from_arrays([vcell.to_numpy(), val["ADEP_mvt"].astype("string").to_numpy()])
    cw = pd.Series(w_cell.reindex(key).to_numpy(), index=val.index)
    n_cell = cw.groupby([vcell.to_numpy(), val["ADEP_mvt"].astype("string").to_numpy()]).transform("size")
    w = (cw / n_cell).fillna(0).to_numpy()
    w = w / w.sum()
    res = pred - yv
    f = pd.DataFrame({"w": w, "wsq": w * res**2, "wres": w * res, "y": yv,
                      "bucket": pd.cut(yv / 60, [-1, 10, 20, 30, 60, np.inf],
                                       labels=["<10", "10-20", "20-30", "30-60", "60+"]),
                      "airport": val["ADEP_mvt"].astype("string").to_numpy(),
                      "hour": val["local_hour"].to_numpy(), "slice": vcell.to_numpy()})
    tot = f["wsq"].sum()
    out = {}
    for by in ("bucket", "airport", "hour", "slice"):
        g = f.groupby(by, observed=True).agg(w=("w", "sum"), wsq=("wsq", "sum"),
                                             wres=("wres", "sum"), n=("y", "size"))
        t = pd.DataFrame({"n": g["n"], "row_share_%": 100 * g["w"],
                          "mse_share_%": 100 * g["wsq"] / tot,
                          "rmse": np.sqrt(g["wsq"] / g["w"]), "bias_s": g["wres"] / g["w"]})
        print(f"\n[{label}] by {by}\n{t.round(1).to_string()}", flush=True)
        t.round(3).reset_index().to_csv(OUT / f"e43_breakdown_{label}_{by}.csv", index=False)
        out[by] = t
    return out


def finish(pred_raw, label, t0, extra=None):
    np.save(OUT / f"e43_pred_{label}.npy", pred_raw)
    pred, _ = apply_regime_model(val, pred_raw, regime)
    pred, _ = apply_extreme_proxy_override(val, pred)
    r = report(yv, pred, val, comp, vcell, trimmed_mask=trimmed)
    row = {"arm": label, "full": r["rmse_full"], "trimmed": r["rmse_trimmed"],
           "board_est": r["rmse_full"] + OFFSET, "sec": round(time.time() - t0), **(extra or {})}
    print(f"{label:<12} full {row['full']:>7.2f}  trimmed {row['trimmed']:>7.2f}  "
          f"board~ {row['board_est']:>7.2f}   ({row['sec']}s)", flush=True)
    with (OUT / "e43_smoke.jsonl" if SMOKE else LOG).open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    return pred


def fit(X_tr, target, X_val, **over):
    m = lgb.LGBMRegressor(**{**PARAMS, **over}, random_state=0).fit(X_tr, target)
    pv = m.predict(X_val)
    del m; gc.collect()
    return pv, None


Xtr, Xval = d.iloc[tr_idx][v5_feats], val[v5_feats]
for arm in ARMS:
    t0 = time.time()
    print(f"\n=== {arm} ===", flush=True)
    if arm == "ref":
        pv, _ = fit(Xtr, ytr, Xval)
        pred = finish(pv, arm, t0)
        breakdown(pred, arm)
    elif arm == "te":
        te = build_te()
        feats = list(te.columns)
        Xtr_te = pd.concat([Xtr, te.iloc[tr_idx]], axis=1)
        Xval_te = pd.concat([Xval, te.iloc[val_idx]], axis=1)
        pv, _ = fit(Xtr_te, ytr, Xval_te)
        pred = finish(pv, arm, t0, {"te_cols": feats})
        breakdown(pred, arm)
        del Xtr_te, Xval_te
    elif arm.startswith("huber"):
        delta = float(arm.removeprefix("huber"))
        pv, _ = fit(Xtr, ytr, Xval, objective="huber", alpha=delta)
        finish(pv, arm, t0, {"delta": delta})
    elif arm.startswith("win"):
        q = float("0." + arm.removeprefix("win"))
        cap = float(np.quantile(ytr, q))
        pv, _ = fit(Xtr, np.minimum(ytr, cap), Xval)
        finish(pv, arm, t0, {"q": q, "cap_s": cap})
    elif arm == "log":
        m = lgb.LGBMRegressor(**PARAMS, random_state=0).fit(Xtr, np.log1p(ytr))
        smear = float(np.mean(np.exp(np.log1p(ytr) - m.predict(Xtr))))  # Duan's estimator
        lp = m.predict(Xval); del m; gc.collect()
        finish(np.expm1(lp), "log_naive", t0)
        finish(np.expm1(lp) * smear, "log_smear", t0, {"smear": smear})
    else:
        raise SystemExit(f"unknown arm {arm}")
