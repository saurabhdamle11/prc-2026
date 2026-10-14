#!/usr/bin/env python
"""E9: where exactly does boosting-from-the-proxy win, and is it robust?

F3 moved the full weighted RMSE 328.0 -> 293.5 while the trimmed number barely moved, so the
whole gain sits in the heavy tail. That is a small number of rows, and a 34 s claim resting on
a few dozen rows needs checking before it is believed.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness import CACHE, load, split
from src.cleaning import trainable_mask
from src.postprocess import fit_regime_branch, apply_regime_branch
from src.validation import ranking_composition, weighted_rmse

S = CACHE
d = load()
is_train, is_val, cell = split(d)
comp = ranking_composition(pd.read_parquet("Dataset/ranking.parquet",
        columns=["PHASE_mvt","MVT_TIME_UTC_mvt","ADEP_mvt"]))
val = d[is_val].copy(); y = val["TAXITIME_SEC_mvt"].to_numpy()
vc, va = cell[is_val], val["ADEP_mvt"]
branch = fit_regime_branch(d[is_train])
noact = val["dq_no_aircraft_type"].to_numpy()
impl = val["dq_implausible_target"].to_numpy()
inv = impl & ~noact
norm = ~(noact | impl)

P = {k: np.load(S/f"pred_{k}.npy") for k in ["F1","F3"] if (S/f"pred_{k}.npy").exists()}
print("population sizes: normal %d  no_actype %d  invisible %d" % (norm.sum(), noact.sum(), inv.sum()))

for k, p in P.items():
    pb, _ = apply_regime_branch(val, p, branch)
    print(f"\n=== {k} ===")
    print(f"  weighted full {weighted_rmse(y, pb, vc, va, comp):7.1f}   "
          f"trimmed {weighted_rmse(y[norm], pb[norm], vc[norm], va[norm], comp):7.1f}")
    for nm, m in [("normal", norm), ("invisible", inv)]:
        print(f"  {nm:<10} plain RMSE {np.sqrt(np.mean((pb[m]-y[m])**2)):9.0f}   "
              f"mean pred {pb[m].mean():8.0f}  mean actual {y[m].mean():8.0f}")

if len(P) == 2:
    a, b = P["F1"], P["F3"]
    ab, _ = apply_regime_branch(val, a, branch); bb, _ = apply_regime_branch(val, b, branch)
    print("\n--- the invisible rows, one by one (F1 -> F3) ---")
    t = pd.DataFrame({"airport": va[inv].to_numpy(), "actual": y[inv],
                      "F1": ab[inv].round(0), "F3": bb[inv].round(0),
                      "proxy_aobt3": val.loc[inv,"proxy_taxi_aobt3"].to_numpy().round(0),
                      "proxy_sched": val.loc[inv,"proxy_taxi_sched"].to_numpy().round(0)})
    t["gain_s2"] = (ab[inv]-y[inv])**2 - (bb[inv]-y[inv])**2
    print(t.sort_values("gain_s2", ascending=False).head(15).to_string(index=False))
    print(f"\ntop-3 rows supply {t.nlargest(3,'gain_s2').gain_s2.sum()/t.gain_s2.sum():.1%} of the invisible-row gain")
    # how much of the full-metric gain survives if the single luckiest row is dropped?
    for drop in [0, 1, 3, 5]:
        idx = np.flatnonzero(inv)[np.argsort(-t.gain_s2.to_numpy())[:drop]]
        keep = np.ones(len(y), bool); keep[idx] = False
        print(f"  dropping the {drop} luckiest invisible rows: "
              f"F1 {weighted_rmse(y[keep],ab[keep],vc[keep],va[keep],comp):7.1f} -> "
              f"F3 {weighted_rmse(y[keep],bb[keep],vc[keep],va[keep],comp):7.1f}")
