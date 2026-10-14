#!/usr/bin/env python
"""E42: does blending BOTH MLPs with LightGBM beat either pairing?

`e39` (overfit, 15 epochs) and `e41` (early-stopped, 8 epochs, 3-seed bag) are the same
architecture trained to different points, and they correlate differently with LightGBM —
0.9003 and 0.9307. The worse standalone model gives the better two-way blend, because its
errors are less like LightGBM's. If the two networks also differ usefully *from each other*,
a three-way blend can beat both pairings.

Free: every prediction set is already cached, so this is arithmetic, no fitting.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.config_features import CONFIG_CATEGORICALS
from src.features import model_features
from src.offblock_gaps import add_offblock_gaps
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)
from src.validation import mirrored_split, ranking_composition, report

OFFSET = 5.73
CONFIG_PREFIXES = ("depcfg_", "arrcfg_", "stand_turnaround", "stand_prev_arr")
CACHE = Path("data/processed/exp_cache")

comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
for extra in ("train_surface", "train_config", "train_osm"):
    d = d.merge(pd.read_parquet(f"data/processed/{extra}.parquet"),
                on="MVT_ID_mvt", how="left", validate="one_to_one")
d, _ = add_offblock_gaps(d)
nums, cats = model_features(d)
cats = list(cats) + [c for c in CONFIG_CATEGORICALS if c in d.columns]
for c in cats:
    d[c] = d[c].astype("string").fillna("__NA__").astype("category")

is_train, is_val, cell = mirrored_split(d)
val = d[is_val].copy()
yv = val["TAXITIME_SEC_mvt"].to_numpy()
vcell, trimmed = cell[is_val], trainable_mask(val)
regime = fit_regime_model(d[is_train], [c for c in nums + cats if not c.startswith(CONFIG_PREFIXES)])

P = {"lgb": np.load(CACHE / "lgb_v5_val_pred.npy"),
     "nn1": np.load(CACHE / "nn_val_pred.npy"),
     "nn2": np.load(CACHE / "nn2_val_pred.npy")}
norm = trimmed.to_numpy()
print("residual correlations on normal rows:")
ks = list(P)
for i, a in enumerate(ks):
    for b in ks[i + 1:]:
        rho = np.corrcoef(P[a][norm] - yv[norm], P[b][norm] - yv[norm])[0, 1]
        print(f"  {a:<4} vs {b:<4} {rho:.4f}")

def score(p, label):
    p = np.asarray(p, dtype="float64").copy()
    p, _ = apply_regime_model(val, p, regime)
    p, _ = apply_extreme_proxy_override(val, p)
    r = report(yv, p, val, comp, vcell, trimmed_mask=trimmed)
    print(f"{label:<38} full {r['rmse_full']:>7.2f}  trimmed {r['rmse_trimmed']:>7.2f}  "
          f"board {r['rmse_full']+OFFSET:>7.2f}")
    return r["rmse_full"], r["rmse_trimmed"]

print()
score(P["lgb"], "LightGBM alone")
score(0.75 * P["lgb"] + 0.25 * P["nn1"], "best 2-way (lgb .75 / nn1 .25)")
score(0.75 * P["lgb"] + 0.25 * P["nn2"], "     2-way (lgb .75 / nn2 .25)")
print()
best = None
for wl in (0.60, 0.65, 0.70, 0.75, 0.80):
    for w1 in np.round(np.arange(0.0, 1.0 - wl + 1e-9, 0.05), 2):
        w2 = round(1.0 - wl - w1, 2)
        if w2 < -1e-9:
            continue
        f, t = score(wl * P["lgb"] + w1 * P["nn1"] + w2 * P["nn2"],
                     f"3-way  lgb {wl:.2f} / nn1 {w1:.2f} / nn2 {w2:.2f}")
        if best is None or f < best[0]:
            best = (f, t, wl, w1, w2)
print(f"\nbest 3-way: lgb {best[2]:.2f} / nn1 {best[3]:.2f} / nn2 {best[4]:.2f} "
      f"-> {best[0]:.2f} full / {best[1]:.2f} trimmed")
print(f"vs LightGBM alone (273.72 / 230.21): {best[0]-273.72:+.2f}s full / {best[1]-230.21:+.2f}s trimmed")
print(f"vs best 2-way     (272.50 / 228.82): {best[0]-272.50:+.2f}s full / {best[1]-228.82:+.2f}s trimmed")
print("\nNOTE: weights chosen on the slice they are scored on, and a 3-way grid searches 60+ "
      "points. Discount accordingly — a broad optimum is believable, a spike is not.")
