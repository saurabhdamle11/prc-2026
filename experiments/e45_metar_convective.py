#!/usr/bin/env python
"""E45: do METAR convective features (IEM, public domain) reduce the error v5 leaves?

`e44` showed July is broadly harder than January and `jul LIRF` dominates. The standard
explanation for summer taxi-out disruption is convective weather -> flow measures -> aircraft
held after pushback. Open-Meteo's gridded `weather_code`/`cape` are already in v5; this adds
what the tower actually reported (`src/metar_features.py`).

Two arms, both on the Oracle VM against e43's `ref` (274.40 / 230.92):

* **sizing** (no fit): v5's residuals split by whether a convective report (TS / VCTS / CB)
  occurred in the 2 h before takeoff — its share of the weighted MSE, and the ceiling if those
  rows were brought down to the non-convective error level of their own (airport, slice) cell.
* **metar**: v5 + the 17 METAR features, one seed, post-processing unchanged.
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
ARMS = sys.argv[1:] or ["sizing", "metar"]

raw = json.loads(Path("data/processed/lgbm_best_params_r3.json").read_text())
INT_P = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}
p = {k: (int(v) if k in INT_P else v) for k, v in raw.items() if not k.startswith("_")}
n_est = p.pop("n_estimators")
SMOKE = "E45_NEST" in os.environ
n_est = int(os.environ.get("E45_NEST", n_est))
PARAMS = dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1, subsample_freq=1,
              n_estimators=n_est, cat_smooth=100.0, **p)
LOG = OUT / ("e45_smoke.jsonl" if SMOKE else "e45.jsonl")

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


for arm in ARMS:
    t0 = time.time()
    print(f"\n=== {arm} ===", flush=True)
    if arm == "sizing":
        pred = post(np.load(OUT / "e43_pred_ref.npy"))
        w, sq, res = row_weights(), (pred - yv) ** 2, pred - yv
        f = pd.DataFrame({"apt": apt_v, "slice": vcell.to_numpy(), "w": w, "wsq": w * sq,
                          "wres": w * res, "conv": val["mt_conv_2h"].to_numpy() > 0,
                          "ts": val["mt_ts_count_6h"].to_numpy() > 0})
        tot = f["wsq"].sum()
        for flag in ("conv", "ts"):
            g = f.groupby(flag).agg(n=("w", "size"), w=("w", "sum"), wsq=("wsq", "sum"),
                                    wres=("wres", "sum"))
            print(f"\nby {flag}: row share, MSE share, RMSE, bias")
            print(pd.DataFrame({"n": g["n"], "row_%": 100 * g["w"], "mse_%": 100 * g["wsq"] / tot,
                                "rmse": np.sqrt(g["wsq"] / g["w"]),
                                "bias": g["wres"] / g["w"]}).round(1).to_string())
        c = f.groupby(["apt", "slice", "conv"]).agg(n=("w", "size"), w=("w", "sum"),
                                                     wsq=("wsq", "sum"), wres=("wres", "sum"))
        c["rmse"], c["bias"] = np.sqrt(c["wsq"] / c["w"]), c["wres"] / c["w"]
        c["mse_%"] = 100 * c["wsq"] / tot
        print("\nby airport x slice x convective-in-2h:")
        print(c[["n", "rmse", "bias", "mse_%"]].round(1).to_string())
        # ceiling: convective rows get the non-convective MSE of their own cell
        base = c.xs(False, level="conv")
        cc = c.xs(True, level="conv")
        new_tot = tot - cc["wsq"].sum() + (cc["w"] * (base["wsq"] / base["w"]).reindex(cc.index)).sum()
        print(f"\nceiling if convective rows matched their cell's non-convective error: "
              f"full {np.sqrt(tot):.2f} -> {np.sqrt(new_tot):.2f}  "
              f"(= {np.sqrt(tot) - np.sqrt(new_tot):.2f}s best case)", flush=True)
        c.round(3).reset_index().to_csv(OUT / "e45_sizing.csv", index=False)
    elif arm == "metar":
        fm = feats + METAR_FEATURES
        m = lgb.LGBMRegressor(**PARAMS, random_state=0).fit(d.iloc[tr_idx][fm], ytr)
        pv = m.predict(val[fm])
        imp = pd.Series(m.booster_.feature_importance("gain"), index=fm)
        rank = imp.rank(ascending=False).astype(int)
        print("METAR feature gain rank (of", len(fm), "):")
        print(pd.DataFrame({"rank": rank[METAR_FEATURES], "gain_%": 100 * imp[METAR_FEATURES] / imp.sum()})
              .sort_values("rank").round(3).to_string())
        del m; gc.collect()
        np.save(OUT / "e45_pred_metar.npy", pv)
        r = report(yv, post(pv), val, comp, vcell, trimmed_mask=trimmed)
        row = {"arm": "metar", "full": r["rmse_full"], "trimmed": r["rmse_trimmed"],
               "board_est": r["rmse_full"] + OFFSET, "sec": round(time.time() - t0)}
        print(f"metar          full {row['full']:>7.2f}  trimmed {row['trimmed']:>7.2f}  "
              f"board~ {row['board_est']:>7.2f}   ({row['sec']}s)   [ref 274.40 / 230.92]", flush=True)
        with LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    else:
        raise SystemExit(f"unknown arm {arm}")
