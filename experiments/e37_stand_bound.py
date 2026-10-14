#!/usr/bin/env python
"""E37: the stand-occupancy bound — a HARD physical cap on taxi time, free from the data.

An aircraft cannot push back from a stand before the previous occupant has vacated it. The
arrival rows are **labelled on both sides** (`TAXITIME_SEC_mvt` is present in
`ranking.parquet`), so the previous occupant's in-block time is known at prediction time:

    in_block(prev arrival) = MVT_TIME(arr) + TAXITIME(arr)
    BLOCK_TIME(this dep)  >  in_block(prev arrival)
    => taxi = MVT_TIME - BLOCK_TIME  <  MVT_TIME - in_block(prev arrival)

That right-hand side is computable for every departure with a mapped stand, and it is a
**hard upper bound**, not a feature. `stand_turnaround_sec` already uses the same join as a
soft feature; this asks whether enforcing it as a *constraint* helps.

Where it should matter most is the missing-aircraft-type regime: those rows are predicted at a
mean of ~7,515 s, and a stand that was occupied 30 minutes before takeoff caps that hard. That
population is 25.6% of the weighted MSE, so even a partial correction is worth more than any
feature measured this round.

Scored on the mirrored protocol using v4's cached predictions — no refit.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.validation import mirrored_split, ranking_composition, weighted_rmse

OFFSET = 5.73
EP = pd.Timestamp("1970-01-01", tz="UTC")

comp = ranking_composition(pd.read_parquet(
    "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
d = pd.read_parquet("data/processed/train_features.parquet")
pred = np.load("data/processed/exp_cache/r4_pred_final.npy")
is_train, is_val, cell = mirrored_split(d)
val = d[is_val].copy()
assert len(pred) == len(val), "cached prediction does not match the split"

# --- the bound: most recent arrival in-block at this (airport, stand) before takeoff -----
raw = pd.read_parquet("Dataset/training_2025-01-01_2026-01-01.parquet",
                      columns=["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt",
                               "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]) \
      if Path("Dataset/training_2025-01-01_2026-01-01.parquet").exists() else None
if raw is None:
    import glob
    raw = pd.concat([pd.read_parquet(f, columns=["PHASE_mvt", "ADEP_mvt", "ADES_mvt",
                                                 "STAND_mvt", "MVT_TIME_UTC_mvt",
                                                 "TAXITIME_SEC_mvt"])
                     for f in sorted(glob.glob("Dataset/training_*.parquet"))],
                    ignore_index=True)
arr = raw[(raw.PHASE_mvt == "ARR") & raw.STAND_mvt.notna() & raw.TAXITIME_SEC_mvt.notna()].copy()
arr["inblock"] = arr.MVT_TIME_UTC_mvt + pd.to_timedelta(arr.TAXITIME_SEC_mvt, unit="s")
arr["stand_key"] = arr.ADES_mvt.astype(str) + "|" + arr.STAND_mvt.astype(str)
arr = arr[["inblock", "stand_key"]].dropna().sort_values("inblock")

left = pd.DataFrame({"t": val.MVT_TIME_UTC_mvt,
                     "stand_key": val.ADEP_mvt.astype(str) + "|" + val.STAND_mvt.astype(str)},
                    index=val.index).sort_values("t")
m = pd.merge_asof(left, arr, left_on="t", right_on="inblock", by="stand_key",
                  direction="backward", allow_exact_matches=False)
m.index = left.index
bound = (m["t"] - m["inblock"]).dt.total_seconds().reindex(val.index)
bound = bound.where(bound.between(60, 86400))          # ignore absurd or negative gaps

yv = val["TAXITIME_SEC_mvt"].to_numpy()
vcell, vapt = cell[is_val], val["ADEP_mvt"]
trim = trainable_mask(val).to_numpy()
noact = val["dq_no_aircraft_type"].to_numpy()
b = bound.to_numpy()

print(f"bound available for {np.isfinite(b).mean():.1%} of validation rows "
      f"({np.isfinite(b[noact]).mean():.1%} of regime rows)")
viol = np.isfinite(b) & (yv > b)
print(f"TRUTH violates the bound on {viol.sum():,} rows ({viol.mean():.3%}) "
      f"-- these are stand reassignments or data errors, and cap how tight it can be")
over = np.isfinite(b) & (pred > b)
print(f"our prediction exceeds the bound on {over.sum():,} rows "
      f"({int((over & noact).sum())} of them regime rows)\n")

def score(p, label):
    full = weighted_rmse(yv, p, vcell, vapt, comp)
    tr_ = weighted_rmse(yv[trim], p[trim], vcell[trim], vapt[trim], comp)
    print(f"{label:<44} full {full:>7.2f}  trimmed {tr_:>7.2f}  board {full+OFFSET:>7.2f}")
    return full

base = score(pred, "v4 (cached, 3-seed)")
for margin in (0, 300, 600):
    p = pred.copy()
    cap = np.isfinite(b) & (p > b + margin)
    p[cap] = b[cap] + margin
    score(p, f"capped at bound + {margin}s  ({int(cap.sum())} rows)")
# regime rows only -- the population this should help most
p = pred.copy()
cap = np.isfinite(b) & noact & (p > b)
p[cap] = b[cap]
score(p, f"cap applied to REGIME rows only ({int(cap.sum())} rows)")
