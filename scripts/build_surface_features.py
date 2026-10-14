#!/usr/bin/env python
"""Compute the rolling surface-state features and cache them keyed on MVT_ID_mvt.

    python scripts/build_surface_features.py

Kept separate from `build_features.py` so the expensive 22 s base build is not repeated every
time a window is retuned; `experiments/` and the training scripts merge the two tables on
`MVT_ID_mvt`. Writes `data/processed/{train,ranking}_surface.parquet`.
"""
from __future__ import annotations

import glob, sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import add_stand_features  # noqa: E402
from src.cleaning import ranking_path  # noqa: E402
from src.surface_state import add_keyed_surface_state, add_surface_state  # noqa: E402

OUT = Path("data/processed")

#: surface_state reads the arrival label and the two off-block plan/actual times on top of
#: what the congestion features already needed.
TRAFFIC_COLS = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
                "AOBT_3_flt", "EOBT_1_flt", "RUNWAY_mvt", "TAXITIME_SEC_mvt", "STAND_mvt"]
DEP_COLS = TRAFFIC_COLS + ["MVT_ID_mvt"]


def run(traffic: pd.DataFrame, label: str) -> None:
    traffic = add_stand_features(traffic)          # stand_area is a grouping key below
    dep = traffic[traffic["PHASE_mvt"] == "DEP"].copy()
    dep["proxy_taxi_aobt3"] = (dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds()
    t0 = time.time()
    out = add_surface_state(dep, traffic)
    # finer-grained realised taxi: same runway, same apron, and the apron->runway pair
    out = add_keyed_surface_state(out, traffic, ("RUNWAY_mvt",), "rwy")
    out = add_keyed_surface_state(out, traffic, ("stand_area",), "area")
    out = add_keyed_surface_state(out, traffic, ("stand_area", "RUNWAY_mvt"), "arearwy")
    keep = ["MVT_ID_mvt"] + [c for c in out.columns if c.startswith("surf_")]
    out[keep].to_parquet(OUT / f"{label}_surface.parquet", index=False)
    print(f"  {label}: {len(out):,} rows, {len(keep)-1} features in {time.time()-t0:.0f}s")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("training...")
    tr = pd.concat([pd.read_parquet(f, columns=DEP_COLS)
                    for f in sorted(glob.glob("Dataset/training_*.parquet"))], ignore_index=True)
    run(tr, "train")
    del tr
    print("ranking...")
    run(pd.read_parquet(ranking_path(), columns=DEP_COLS), "ranking")


if __name__ == "__main__":
    main()
