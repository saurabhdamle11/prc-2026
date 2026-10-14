#!/usr/bin/env python
"""E31: how accurate must a self-computed off-block be to reach the leader's 224 s?

The leader is at **224 s**. v4's *trimmed* weighted RMSE -- every anomalous row solved
perfectly -- is 231.49 offline, i.e. 237.2 on the board. So the tail cannot close this gap at
all, and the normal-row model has to improve by 13-41 s. That requires information we do not
have, and the prime suspect is an off-block time computed from raw ADS-B rather than the
supplied `AOBT_3_flt`.

Before spending the remaining window on that, quantify it. The identity is exact:

    TAXITIME = MVT_TIME - BLOCK_TIME

so **an off-block estimator with RMS error `s` yields a taxi predictor with RMS error `s`**,
with no modelling in between. That makes the target directly computable: to reach a normal-row
weighted RMSE of `R`, the detector must be accurate to roughly `R` seconds.

Training rows carry `BLOCK_TIME_UTC_mvt`, so the error of every *existing* off-block estimate
can be measured against the truth right now, on the mirrored protocol's own validation cells
and weighting. That gives both the bar the detector must clear and the headroom over the
proxy the model is currently leaning on.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.validation import mirrored_split, ranking_composition, weighted_rmse

OFFSET = 5.73          # measured board-minus-offline (v3 +5.61, v4 +5.73)
V4_TRIM = 231.49       # v4 normal-row weighted RMSE, from e30

comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
is_train, is_val, cell = mirrored_split(d)
val = d[is_val].copy()
normal = trainable_mask(val).to_numpy()
V = val[normal]
vcell, vapt = cell[is_val][normal], V["ADEP_mvt"]
y = V["TAXITIME_SEC_mvt"].to_numpy()

print(f"normal validation rows: {len(V):,}   (v4's model scores {V4_TRIM:.2f} on these)\n")
print("=== error of each off-block estimate, measured against the true BLOCK_TIME ===")
print("An estimator with RMS error s gives a taxi predictor with RMS error s -- the identity")
print("TAXITIME = MVT_TIME - BLOCK_TIME leaves no room for anything else.\n")
print(f"{'off-block estimate':<26} {'coverage':>9} {'weighted RMSE':>14} {'median bias':>12} {'p90 |err|':>10}")
ests = {"AOBT_3_flt (ADS-B, used now)": "AOBT_3_flt", "EOBT_1_flt (filed est.)": "EOBT_1_flt",
        "LOBT_flt (last off-block)": "LOBT_flt", "IOBT_flt (initial off-block)": "IOBT_flt",
        "SCHED_TIME_UTC_mvt": "SCHED_TIME_UTC_mvt"}
for label, col in ests.items():
    err = (V[col] - V["BLOCK_TIME_UTC_mvt"]).dt.total_seconds()
    m = err.notna().to_numpy()
    if m.sum() == 0:
        continue
    # the taxi predictor implied by this estimate is MVT - estimate; its error is exactly `err`
    pred = (V["MVT_TIME_UTC_mvt"] - V[col]).dt.total_seconds().to_numpy()
    r = weighted_rmse(y[m], pred[m], vcell[m], vapt[m], comp)
    print(f"{label:<26} {m.mean():>8.1%} {r:>14.1f} {err[m].median():>12.0f} "
          f"{err[m].abs().quantile(0.9):>10.0f}")

print(f"\n{'the model (v4), combining all of them':<26} {'100.0%':>8} {V4_TRIM:>14.1f}")

print("\n=== so what detector accuracy reaches which board score? ===")
print("Assumes the model can exploit a better off-block about as well as it exploits the")
print("current one -- optimistic, but it brackets the answer.\n")
print(f"{'detector RMS error':>20} {'normal-row RMSE':>17} {'board (anomalies as now)':>26}")
mse_v4_full = 276.50 ** 2
anom_mse = mse_v4_full * (1 - 0.706)          # e19: normal rows are 70.6% of the weighted MSE
for s in (400, 300, 250, 231, 200, 175, 150, 120):
    board = np.sqrt(s ** 2 + anom_mse) + OFFSET
    flag = "  <- beats 224" if board <= 224 else ""
    print(f"{s:>20} {s:>17} {board:>26.1f}{flag}")

print("\n=== and if the anomalous rows were ALSO solved ===")
for s in (300, 250, 231, 218, 200, 175, 150):
    print(f"{s:>20} {s:>17} {s + OFFSET:>26.1f}"
          f"{'  <- beats 224' if s + OFFSET <= 224 else ''}")
