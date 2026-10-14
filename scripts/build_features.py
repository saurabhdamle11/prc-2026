#!/usr/bin/env python
"""Build the modelling feature table for training and ranking rows, and cache it.

    python scripts/build_features.py

Writes `data/processed/{train,ranking}_features.parquet`. Both sides run through the exact
same function chain, which is the point: the ranking set is featurised with its own traffic
as the congestion universe, never with training traffic.

The congestion universe only needs six columns, so it is loaded separately from the full DEP
frame — that keeps peak memory to roughly the size of the DEP rows alone rather than the
whole 4.2M-row file with all 30 columns.
"""

from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning import add_offblock_proxies, clean_movements  # noqa: E402
from src.cleaning import ranking_path  # noqa: E402
from src.external_data import (  # noqa: E402
    attach_holidays,
    attach_weather,
    load_holidays,
    load_weather,
)
from src.features import build_features  # noqa: E402

OUT_DIR = Path("data/processed")

#: The only columns the congestion/queue/sequence features read from the traffic universe.
TRAFFIC_COLS = [
    "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "RUNWAY_mvt",
]


def _featurise(dep: pd.DataFrame, traffic: pd.DataFrame, weather, hol) -> pd.DataFrame:
    dep = add_offblock_proxies(clean_movements(dep, phase="DEP"))
    dep = attach_weather(dep, weather)
    dep = attach_holidays(dep, hol)
    return build_features(dep, traffic)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weather, hol = load_weather(), load_holidays()

    train_files = sorted(glob.glob("Dataset/training_*.parquet"))

    print("loading training traffic universe...")
    traffic = pd.concat([pd.read_parquet(f, columns=TRAFFIC_COLS) for f in train_files],
                        ignore_index=True)
    print(f"  {len(traffic):,} movements")

    print("loading training DEP rows...")
    dep = pd.concat(
        [pd.read_parquet(f).query("PHASE_mvt == 'DEP'") for f in train_files],
        ignore_index=True,
    )
    print(f"  {len(dep):,} DEP rows")

    t0 = time.time()
    feats = _featurise(dep, traffic, weather, hol)
    print(f"  featurised in {time.time() - t0:.0f}s -> {feats.shape}")
    feats.to_parquet(OUT_DIR / "train_features.parquet", index=False)
    del dep, traffic, feats

    print("\nranking set...")
    rank_traffic = pd.read_parquet(ranking_path(), columns=TRAFFIC_COLS)
    rank = pd.read_parquet(ranking_path())
    n_dep = int((rank["PHASE_mvt"] == "DEP").sum())
    rfeats = _featurise(rank, rank_traffic, weather, hol)
    assert len(rfeats) == n_dep, "ranking rows lost — submission would be rejected"
    print(f"  {rfeats.shape}, all {n_dep:,} DEP rows preserved")
    rfeats.to_parquet(OUT_DIR / "ranking_features.parquet", index=False)

    print("\ndone.")


if __name__ == "__main__":
    main()
