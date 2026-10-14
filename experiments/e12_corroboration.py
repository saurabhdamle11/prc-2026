#!/usr/bin/env python
"""E12: size the corroboration rule on the 1-2 h band, where the ranking exposure is 180x larger.

The >2 h band is 5 ranking rows. The [3600, 7200) band is 904, so if agreement among the
off-block estimates identifies a trustworthy proxy there too, it is worth far more.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

P = ["proxy_taxi_sched","proxy_taxi_eobt1","proxy_taxi_iobt","proxy_taxi_lobt"]
cols = ["TAXITIME_SEC_mvt","proxy_taxi_aobt3","ADEP_mvt","dq_no_aircraft_type"] + P
tr = pd.read_parquet("data/processed/train_features.parquet", columns=cols)
tr = tr[tr.TAXITIME_SEC_mvt.notna()]
tr["n_agree"] = (tr[P].sub(tr.proxy_taxi_aobt3, axis=0).abs() < 1800).sum(axis=1)

def rmse(p, y): return float(np.sqrt(np.mean((np.asarray(p, float) - y) ** 2)))

for lo, hi in [(1800,3600),(3600,7200),(7200,10**9)]:
    s = tr[tr.proxy_taxi_aobt3.between(lo, hi, inclusive="left")]
    if len(s) == 0: continue
    y = s.TAXITIME_SEC_mvt.to_numpy()
    print(f"\n=== proxy in [{lo}, {hi}) : n={len(s):,} ===")
    print(f"{'n_agree':>8}{'n':>8}{'E[y]':>9}{'med y':>8}{'med proxy':>11}{'|y-proxy|<900':>15}")
    for k in range(5):
        m = s.n_agree == k
        if m.sum() == 0: continue
        t = s[m]
        print(f"{k:>8}{len(t):>8,}{t.TAXITIME_SEC_mvt.mean():>9,.0f}{t.TAXITIME_SEC_mvt.median():>8,.0f}"
              f"{t.proxy_taxi_aobt3.median():>11,.0f}"
              f"{((t.TAXITIME_SEC_mvt-t.proxy_taxi_aobt3).abs()<900).mean():>15.1%}")
    # is a proxy-trusting override better than the group mean, conditional on agreement?
    for k in [2, 3, 4]:
        m = (s.n_agree >= k).to_numpy()
        if m.sum() < 5: continue
        print(f"  rows with n_agree>={k} (n={m.sum():,}): "
              f"RMSE(group mean) {rmse(np.full(m.sum(), s.TAXITIME_SEC_mvt[m].mean()), y[m]):>7,.0f}   "
              f"RMSE(proxy) {rmse(s.proxy_taxi_aobt3.to_numpy()[m], y[m]):>7,.0f}")

r = pd.read_parquet("data/processed/ranking_features.parquet",
                    columns=["proxy_taxi_aobt3","ADEP_mvt"] + P)
r["n_agree"] = (r[P].sub(r.proxy_taxi_aobt3, axis=0).abs() < 1800).sum(axis=1)
print("\n\nRANKING exposure by band x agreement:")
r["band"] = pd.cut(r.proxy_taxi_aobt3, [-1e9,1800,3600,7200,1e9],
                   labels=["<1800","1800-3600","3600-7200",">7200"])
print(pd.crosstab(r.band, r.n_agree).to_string())
