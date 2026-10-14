#!/usr/bin/env python
"""Build METAR features for both sides -> data/processed/{train,ranking}_metar.parquet.

    python scripts/fetch_metar.py          # once
    python scripts/build_metar_features.py

Keyed by MVT_ID_mvt, merged one-to-one like the surface/config/OSM side tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metar_features import METAR_FEATURES, build_metar_features  # noqa: E402

COLS = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]


def main() -> None:
    raw = pd.read_parquet("data/external/metar_raw.parquet")
    for side in ("train", "ranking"):
        fl = pd.read_parquet(f"data/processed/{side}_features.parquet", columns=COLS)
        out = build_metar_features(fl, raw)
        assert len(out) == len(fl) and out["MVT_ID_mvt"].is_unique
        out.to_parquet(f"data/processed/{side}_metar.parquet", index=False)
        cover = out["mt_vis_m"].notna().mean() * 100
        print(f"{side}: {len(out):,} rows, report matched for {100 - out['mt_reports_1h'].isna().mean()*100:.2f}%, "
              f"vis parsed {cover:.1f}%")
        print(out[METAR_FEATURES].mean().round(4).to_string())


if __name__ == "__main__":
    main()
