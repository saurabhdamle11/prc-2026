#!/usr/bin/env python
"""Exact error budget for the mirrored protocol: which row populations carry the MSE.

Reuses the saved lgbm_mirrored.txt booster so no refit is needed.
"""
from __future__ import annotations
import sys
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness import CACHE
from src.cleaning import trainable_mask
from src.features import model_features
from src.postprocess import fit_regime_branch, apply_regime_branch
from src.validation import mirrored_split, ranking_composition, weighted_rmse

comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
nums, cats = model_features(d)
for c in cats:
    d[c] = d[c].astype("string").fillna("__NA__").astype("category")
feats = nums + cats
is_train, is_val, cell = mirrored_split(d)
val = d[is_val].copy()
yv = val["TAXITIME_SEC_mvt"].to_numpy()

booster = lgb.Booster(model_file="data/processed/lgbm_mirrored.txt")
pred = booster.predict(val[feats])
np.save(CACHE / "base_pred.npy", pred)

branch = fit_regime_branch(d[is_train])
pred_b, n = apply_regime_branch(val, pred, branch)
print(f"branch fitted on {branch['n_fitted']} train rows; applied to {n} val rows")
print(f"  slope {branch['slope']:.4f} intercept {branch['intercept']:.0f}")

vcell, vapt = cell[is_val], val["ADEP_mvt"]
def wr(p, m=None):
    if m is None:
        return weighted_rmse(yv, p, vcell, vapt, comp)
    return weighted_rmse(yv[m], p[m], vcell[m], vapt[m], comp)

noact = val["dq_no_aircraft_type"].to_numpy()
impl  = val["dq_implausible_target"].to_numpy()
invisible = impl & ~noact                      # broken label, NO visible signature
normal = ~(noact | impl)

print(f"\nval rows {len(val):,}   no_actype {noact.sum()}   invisible-implausible {invisible.sum()}   normal {normal.sum():,}")
print(f"\nRMSE (weighted, full set)")
print(f"  no branch                 {wr(pred):8.1f}")
print(f"  constant-mean branch      {wr(np.where(noact, d.loc[is_train & d.dq_no_aircraft_type,'TAXITIME_SEC_mvt'].mean(), pred)):8.1f}")
print(f"  linear branch (shipped)   {wr(pred_b):8.1f}")

# oracle experiments: replace each population with the truth, one at a time
for name, m in [("no_actype", noact), ("invisible-implausible", invisible), ("both", noact | impl)]:
    p = pred_b.copy(); p[m] = yv[m]
    print(f"  oracle on {name:<22}{wr(p):8.1f}")
p = pred_b.copy(); p[normal] = yv[normal]
print(f"  oracle on normal rows     {wr(p):8.1f}")

print(f"\n  trimmed (normal only)     {wr(pred_b, trainable_mask(val).to_numpy()):8.1f}")

# MSE share per population, on the leaderboard weighting
err = pd.DataFrame({"slice": vcell.to_numpy(), "airport": vapt.to_numpy(),
                    "sq": (pred_b - yv) ** 2,
                    "pop": np.where(noact, "no_actype", np.where(impl, "invisible", "normal"))})
cellw = comp.set_index(["slice", "airport"])["weight"]
tot = err.groupby(["slice", "airport"], observed=True)["sq"].mean()
print("\nweighted-MSE share by population")
for pop, g in err.groupby("pop"):
    # contribution of this population to each cell's mean square, times cell weight
    contrib = (g.groupby(["slice", "airport"], observed=True)["sq"].sum()
               / err.groupby(["slice", "airport"], observed=True)["sq"].size())
    w = cellw.reindex(contrib.index).fillna(0); w = w / cellw.reindex(tot.index).fillna(0).sum()
    share = (w * contrib).sum() / (cellw.reindex(tot.index).fillna(0) / cellw.reindex(tot.index).fillna(0).sum() * tot).sum()
    print(f"  {pop:<12} n={len(g):>7,}  share of weighted MSE {share:6.1%}")
