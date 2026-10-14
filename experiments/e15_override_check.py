#!/usr/bin/env python
"""E15: is the midpoint of (proxy, plan consensus) the right value for the override?"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.postprocess import EXTREME_PROXY_SEC, MIN_AGREE

d = pd.read_parquet("data/processed/train_features.parquet",
    columns=["TAXITIME_SEC_mvt","proxy_taxi_aobt3","offblock_agree_n","offblock_plan_median","ADEP_mvt"])
d = d[d.TAXITIME_SEC_mvt.notna()]
fire = d[(d.proxy_taxi_aobt3 > EXTREME_PROXY_SEC) & (d.offblock_agree_n >= MIN_AGREE)]
skip = d[(d.proxy_taxi_aobt3 > EXTREME_PROXY_SEC) & (d.offblock_agree_n < MIN_AGREE)]
y = fire.TAXITIME_SEC_mvt.to_numpy()
rmse = lambda p: float(np.sqrt(np.mean((np.asarray(p, float) - y) ** 2)))
print(f"rule fires on {len(fire)} training rows, abstains on {len(skip)}")
print(f"  abstained rows' targets: min {skip.TAXITIME_SEC_mvt.min():,.0f} max {skip.TAXITIME_SEC_mvt.max():,.0f} "
      f"median {skip.TAXITIME_SEC_mvt.median():,.0f}  -> abstaining is correct\n")
P, M = fire.proxy_taxi_aobt3.to_numpy(), fire.offblock_plan_median.to_numpy()
for name, v in [("proxy only", P), ("plan consensus only", M), ("midpoint (shipped)", (P+M)/2),
                ("0.75*proxy+0.25*plan", .75*P+.25*M), ("0.25*proxy+0.75*plan", .25*P+.75*M),
                ("predict ~1000s instead", np.full(len(fire), 1000.0))]:
    print(f"  {name:<26} RMSE {rmse(v):>9,.0f}")
