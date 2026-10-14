#!/usr/bin/env python
"""E11: when AOBT_3 is implausibly early, is the whole record shifted or just that field?

If a movement carries a date error, every timestamp derived from that record moves together,
and BLOCK_TIME moves with them -- so the proxy is telling the truth and should be trusted. If
only AOBT_3 is out, the proxy is garbage and the row is an ordinary taxi.

The two cases are distinguishable without the label: compare AOBT_3 against the *other*
off-block estimates on the same row (SCHED / EOBT_1 / IOBT / LOBT). Agreement means the whole
record is shifted; disagreement means AOBT_3 alone is broken.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cols = ["TAXITIME_SEC_mvt","proxy_taxi_aobt3","proxy_taxi_sched","proxy_taxi_eobt1",
        "proxy_taxi_iobt","proxy_taxi_lobt","ADEP_mvt","MVT_ID_mvt"]
tr = pd.read_parquet("data/processed/train_features.parquet", columns=cols)
hi = tr[(tr.proxy_taxi_aobt3 > 7200) & tr.TAXITIME_SEC_mvt.notna()].copy()

def agree(df):
    """Does AOBT_3 sit near the plan-side off-block times, or far from all of them?"""
    others = df[["proxy_taxi_sched","proxy_taxi_eobt1","proxy_taxi_iobt","proxy_taxi_lobt"]]
    return (others.sub(df.proxy_taxi_aobt3, axis=0).abs() < 1800).sum(axis=1), others.median(axis=1)

hi["n_agree"], hi["other_med"] = agree(hi)
hi["y_near_proxy"] = (hi.TAXITIME_SEC_mvt - hi.proxy_taxi_aobt3).abs() < 900
pd.set_option("display.width", 250)
print("TRAINING rows with proxy_taxi_aobt3 > 7200 (n=%d)\n" % len(hi))
print(hi[["ADEP_mvt","TAXITIME_SEC_mvt","proxy_taxi_aobt3","proxy_taxi_sched","proxy_taxi_eobt1",
          "other_med","n_agree","y_near_proxy"]].sort_values("proxy_taxi_aobt3").round(0).to_string(index=False))

print("\nrule: trust the proxy only when >=2 other off-block estimates agree with it")
for k in [0,1,2,3,4]:
    m = hi.n_agree >= k
    if m.sum() == 0: continue
    print(f"  n_agree >= {k}: fires on {m.sum():>3}/{len(hi)} rows, "
          f"proxy correct (|y-proxy|<900) on {hi.loc[m,'y_near_proxy'].mean():>6.1%}"
          f"   | not fired: correct on {hi.loc[~m,'y_near_proxy'].mean() if (~m).any() else float('nan'):>6.1%}")

def rmse(p, y): return float(np.sqrt(np.mean((p - y) ** 2)))
y = hi.TAXITIME_SEC_mvt.to_numpy()
base = np.full(len(hi), 1000.0)
print(f"\n  predict normal (1000s) everywhere : RMSE {rmse(base, y):>9,.0f}")
print(f"  predict the proxy everywhere      : RMSE {rmse(hi.proxy_taxi_aobt3.to_numpy(), y):>9,.0f}")
for k in [1,2,3]:
    p = np.where(hi.n_agree >= k, hi.proxy_taxi_aobt3, 1000.0)
    print(f"  trust proxy only when n_agree>={k}  : RMSE {rmse(p, y):>9,.0f}")

print("\n" + "="*70)
r = pd.read_parquet("data/processed/ranking_features.parquet", columns=cols[:-1] + ["MVT_ID_mvt"])
rh = r[r.proxy_taxi_aobt3 > 7200].copy()
rh["n_agree"], rh["other_med"] = agree(rh)
print(f"RANKING rows with proxy > 7200 (n={len(rh)}) -- what the rule would do:\n")
print(rh[["ADEP_mvt","proxy_taxi_aobt3","proxy_taxi_sched","proxy_taxi_eobt1","proxy_taxi_iobt",
          "other_med","n_agree"]].sort_values("proxy_taxi_aobt3").round(0).to_string(index=False))
