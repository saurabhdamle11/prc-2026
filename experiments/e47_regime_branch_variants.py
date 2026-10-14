#!/usr/bin/env python
"""E47: can the missing-aircraft-type branch itself be improved? (placeholders, loss, peer delay)

The branch (`src/postprocess.py::fit_regime_model`) is a small LightGBM fitted on every
missing-type training row, target ``proxy_taxi_sched - y``. A placeholder row
(BLOCK == SCHED) has ``y == proxy_taxi_sched``, so it enters as an exact-zero target. Variants:

* ``no_ph_exact`` / ``no_ph_60`` — train without placeholders (exact, or |BLOCK-SCHED| <= 60 s);
* ``huber600`` / ``huber1800`` — Huber on the delta, inside the branch only;
* ``log_smear`` — fit log1p(y) directly, Duan-smeared back (drops the delta form);
* ``peer`` — add peer-disruption features from `_mvt` columns only: stats of
  ``MVT_TIME - SCHED_TIME`` over the *other* departures at the same airport scheduled within
  +-60 min (mean, median, share > 1 h late, count). Symmetric window, like the congestion
  features; both columns are 100% present in ranking.

Scored on the branch's own rows by leave-one-month-out over all 12 months (fit on the other
11 months' missing-type rows, predict the held-out month), because a single holdout of ~400
rows dominated by a few 10 h labels is noise. Also scored on the mirrored protocol total,
with e43's main-model predictions for every other row (ref 274.40).
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
from src.postprocess import (_REGIME_PARAMS, apply_extreme_proxy_override, apply_regime_model,
                             fit_regime_model)
from src.validation import mirrored_split, ranking_composition, report

OFFSET = 5.73
CONFIG_PREFIXES = ("depcfg_", "arrcfg_", "stand_turnaround", "stand_prev_arr")
OUT = Path("data/processed/exp_cache")
ARMS = sys.argv[1:] or ["regime"]

raw = json.loads(Path("data/processed/lgbm_best_params_r3.json").read_text())
INT_P = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}
p = {k: (int(v) if k in INT_P else v) for k, v in raw.items() if not k.startswith("_")}
n_est = p.pop("n_estimators")
SMOKE = "E47_NEST" in os.environ
n_est = int(os.environ.get("E47_NEST", n_est))
PARAMS = dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1, subsample_freq=1,
              n_estimators=n_est, cat_smooth=100.0, **p)
LOG = OUT / ("e47_smoke.jsonl" if SMOKE else "e47.jsonl")

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



# ---- peer-disruption features over ALL departures (train file is the full DEP record) -----
def peer_features(df):
    s = df["SCHED_TIME_UTC_mvt"]
    late = (df["MVT_TIME_UTC_mvt"] - s).dt.total_seconds()
    out = pd.DataFrame(index=df.index, columns=["peer_late_mean", "peer_late_median",
                                                "peer_late_gt1h", "peer_n"], dtype="float64")
    ok = s.notna() & late.notna()
    for a, idx in df[ok].groupby(df.loc[ok, "ADEP_mvt"].astype("string")).groups.items():
        g = pd.DataFrame({"t": s.loc[idx].dt.tz_localize(None) if s.dt.tz is not None else s.loc[idx],
                          "x": late.loc[idx]}).sort_values("t")
        r = g.set_index("t")["x"]
        big = (r > 3600).astype("float64")
        win = pd.Timedelta("120min")
        n = r.rolling(win, center=True).count().to_numpy()
        sm = r.rolling(win, center=True).sum().to_numpy()
        med = r.rolling(win, center=True).median().to_numpy()
        gt = big.rolling(win, center=True).sum().to_numpy()
        x, b = r.to_numpy(), big.to_numpy()
        denom = np.where(n > 1, n - 1, np.nan)
        out.loc[g.index, "peer_late_mean"] = (sm - x) / denom
        out.loc[g.index, "peer_late_median"] = med  # includes self; ~40 peers so negligible
        out.loc[g.index, "peer_late_gt1h"] = (gt - b) / denom
        out.loc[g.index, "peer_n"] = n - 1
    return out

t0 = time.time()
PEER = ["peer_late_mean", "peer_late_median", "peer_late_gt1h", "peer_n"]
d[PEER] = peer_features(d)
val[PEER] = d.iloc[val_idx][PEER].to_numpy()
print(f"peer features built in {time.time()-t0:.0f}s; missing-type rows with peer_late_gt1h > 0: "
      f"{(d.loc[d.dq_no_aircraft_type, 'peer_late_gt1h'] > 0).mean()*100:.1f}%", flush=True)

base_cols = [c for c in feats if not c.startswith(CONFIG_PREFIXES)]
month = d["MVT_TIME_UTC_mvt"].dt.month
A = d[d["dq_no_aircraft_type"] & d["TAXITIME_SEC_mvt"].notna()].copy()
A["m"] = month[A.index]
bs = (A["BLOCK_TIME_UTC_mvt"] - A["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
A["ph_exact"], A["ph_60"] = (bs == 0).to_numpy(), (bs.abs() <= 60).to_numpy()


def fit_branch(tr, variant):
    cols = base_cols + (PEER if variant == "peer" else [])
    usable = [c for c in cols if tr[c].notna().mean() > 0.5]
    if variant == "no_ph_exact":
        tr = tr[~tr["ph_exact"]]
    elif variant == "no_ph_60":
        tr = tr[~tr["ph_60"]]
    params = dict(_REGIME_PARAMS)
    if variant.startswith("huber"):
        params.update(objective="huber", alpha=float(variant.removeprefix("huber")))
    y_ = tr["TAXITIME_SEC_mvt"]
    if variant == "log_smear":
        tgt = np.log1p(y_.clip(lower=0))
    else:
        tgt = tr["proxy_taxi_sched"] - y_
    ok = tgt.notna()
    m = lgb.LGBMRegressor(random_state=0, n_jobs=-1, **params).fit(tr.loc[ok, usable], tgt[ok])
    smear = float(np.mean(np.exp(tgt[ok] - m.predict(tr.loc[ok, usable])))) if variant == "log_smear" else 1.0
    return {"m": m, "cols": usable, "variant": variant, "smear": smear,
            "const": float(y_.mean())}


def predict_branch(b, df):
    raw = b["m"].predict(df[b["cols"]])
    if b["variant"] == "log_smear":
        v = np.expm1(raw) * b["smear"]
    else:
        v = df["proxy_taxi_sched"].to_numpy("float64") - raw
    return np.maximum(np.where(np.isfinite(v), v, b["const"]), 0.0)


VARIANTS = ["base", "no_ph_exact", "no_ph_60", "huber600", "huber1800", "log_smear", "peer"]
ref_raw = np.load(OUT / "e43_pred_ref.npy")
w = row_weights()
na_v = val["dq_no_aircraft_type"].to_numpy()
rows, per_month = [], {}
for v in VARIANTS:
    sq_all, per = [], {}
    for mth in range(1, 13):
        tr, te = A[A["m"] != mth], A[A["m"] == mth]
        if len(te) == 0:
            continue
        p_ = predict_branch(fit_branch(tr, v), te)
        e = (p_ - te["TAXITIME_SEC_mvt"].to_numpy()) ** 2
        sq_all.append(e); per[mth] = np.sqrt(e.mean())
    lomo = float(np.sqrt(np.concatenate(sq_all).mean()))
    # mirrored protocol: fit on train months, apply to validation missing-type rows
    b = fit_branch(A[A.index.isin(d.index[is_train])], v)
    full = ref_raw.copy()
    full[na_v] = predict_branch(b, val[na_v])
    full, _ = apply_extreme_proxy_override(val, full)
    tot = np.sqrt((w * (full - yv) ** 2).sum())
    r = report(yv, full, val, comp, vcell, trimmed_mask=trimmed)
    rows.append({"variant": v, "lomo_rmse": lomo, "lomo_jul": per.get(7), "mirrored_full": r["rmse_full"]})
    per_month[v] = per
    print(f"{v:<12} LOMO {lomo:7.0f}   Jul {per.get(7, np.nan):7.0f}   mirrored full {r['rmse_full']:7.2f}",
          flush=True)

pm = pd.DataFrame(per_month).round(0)
base = pm["base"]
print("\nLOMO RMSE by held-out month (rows):")
pm.insert(0, "n", A.groupby("m").size())
print(pm.to_string())
print("\nmonths beating base:", {v: int((pm[v] < base).sum()) for v in VARIANTS if v != "base"})
res = pd.DataFrame(rows)
res.round(2).to_csv(OUT / "e47_regime_variants.csv", index=False)
with LOG.open("a") as fh:
    for r_ in rows:
        fh.write(json.dumps(r_) + "\n")
