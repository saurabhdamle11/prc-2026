#!/usr/bin/env python
"""E17: where does the error sit AFTER round 2? Same decomposition as budget.py, new model."""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness import CACHE, load, split, featurelist, encode
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)
from src.validation import ranking_composition, weighted_rmse

S = CACHE
d = load()
is_train, is_val, cell = split(d)
nums, cats = featurelist(d, drop={"stand_rwy"}, extra_cats=())
d = encode(d, cats); feats = nums + cats
comp = ranking_composition(pd.read_parquet("Dataset/ranking.parquet",
        columns=["PHASE_mvt","MVT_TIME_UTC_mvt","ADEP_mvt"]))
val = d[is_val]; y = val["TAXITIME_SEC_mvt"].to_numpy()
vc, va = cell[is_val], val["ADEP_mvt"]
pred = np.load(S/"final_p_surface_p_corroborate.npy")
p, _ = apply_regime_model(val, pred, fit_regime_model(d[is_train], feats))
p, _ = apply_extreme_proxy_override(val, p)

noact = val["dq_no_aircraft_type"].to_numpy()
impl = val["dq_implausible_target"].to_numpy()
inv = impl & ~noact
norm = ~(noact | impl)
W = lambda m=None: (weighted_rmse(y, p, vc, va, comp) if m is None
                    else weighted_rmse(y[m], p[m], vc[m], va[m], comp))
print(f"final weighted RMSE {W():.1f}   trimmed {W(norm):.1f}\n")
for nm, m in [("normal", norm), ("no_actype", noact), ("invisible", inv)]:
    q = p.copy(); q[m] = y[m]
    print(f"  oracle on {nm:<10} ({m.sum():>6,} rows) -> {weighted_rmse(y,q,vc,va,comp):7.1f}  "
          f"(headroom {W()-weighted_rmse(y,q,vc,va,comp):6.1f}s)")
print("\nper-cell, worst first:")
e = pd.DataFrame({"slice": vc.to_numpy(), "airport": va.to_numpy(), "sq": (p-y)**2})
c = (e.groupby(["slice","airport"], observed=True)["sq"].agg(n="size", mse="mean")
       .reset_index().merge(comp[["slice","airport","weight"]], on=["slice","airport"]))
c["rmse"] = np.sqrt(c.mse); c["share"] = c.weight*c.mse/(c.weight*c.mse).sum()
print(c.sort_values("share", ascending=False)[["slice","airport","n","rmse","weight","share"]]
       .head(8).round(3).to_string(index=False))
