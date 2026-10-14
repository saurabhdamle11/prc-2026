#!/usr/bin/env python
"""E16: could the override rule have been derived from the training window alone?

Its two constants (a 7200 s threshold, >=2 corroborating estimates) were picked from a table
covering all 12 months, which includes the validation months. If the same choice falls out of
Jan-Jun alone, the 34.8 s measured on Dec+Jul is an honest out-of-sample number; if not, it is
partly circular and should be quoted with that caveat.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

d = pd.read_parquet("data/processed/train_features.parquet",
    columns=["TAXITIME_SEC_mvt","proxy_taxi_aobt3","offblock_agree_n","offblock_plan_median",
             "ADEP_mvt","MVT_TIME_UTC_mvt"])
d = d[d.TAXITIME_SEC_mvt.notna()]
m = d.MVT_TIME_UTC_mvt.dt.month
tr, va = d[m.isin([1,2,3,4,5,6])], d[m.isin([7,12])]

for nm, s in [("TRAIN WINDOW (Jan-Jun)", tr), ("VALIDATION (Jul + Dec)", va)]:
    hi = s[s.proxy_taxi_aobt3 > 7200]
    print(f"\n{nm}: {len(hi)} rows with proxy > 7200")
    if not len(hi): continue
    print(hi[["ADEP_mvt","TAXITIME_SEC_mvt","proxy_taxi_aobt3","offblock_plan_median","offblock_agree_n"]]
          .sort_values("offblock_agree_n").round(0).to_string(index=False))
    f = hi[hi.offblock_agree_n >= 2]; a = hi[hi.offblock_agree_n < 2]
    print(f"  fires on {len(f)} (targets {f.TAXITIME_SEC_mvt.min():,.0f}-{f.TAXITIME_SEC_mvt.max():,.0f}), "
          f"abstains on {len(a)} (targets {a.TAXITIME_SEC_mvt.min():,.0f}-{a.TAXITIME_SEC_mvt.max():,.0f})"
          if len(f) and len(a) else f"  fires on {len(f)}, abstains on {len(a)}")
    if len(f):
        y = f.TAXITIME_SEC_mvt.to_numpy()
        mid = (f.proxy_taxi_aobt3.to_numpy() + f.offblock_plan_median.to_numpy()) / 2
        r = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
        print(f"  on fired rows: midpoint RMSE {r(mid):,.0f}  vs predicting 1,000 s: {r(np.full(len(f),1000.0)):,.0f}")
