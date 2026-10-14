#!/usr/bin/env python
"""Attach OpenStreetMap taxi geometry to both feature tables.

    python scripts/fetch_osm_airports.py       # once; writes data/external/osm/*.json
    python scripts/build_osm_features.py       # -> data/processed/{train,ranking}_osm.parquet

See `src/osm_features.py` for what the features mean and `SKILLS.md` §11 for what they are
worth. Attribution: "Airport geometry (c) OpenStreetMap contributors, licensed under ODbL 1.0."
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.osm_features import OSM_FEATURES, add_osm_features  # noqa: E402

COLS = ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]

for side in ("train", "ranking"):
    src = Path(f"data/processed/{side}_features.parquet")
    if not src.exists():
        raise SystemExit(f"{src} not found — run scripts/build_features.py first")
    t0 = time.time()
    q = pd.read_parquet(src, columns=COLS)
    out = add_osm_features(q)[["MVT_ID_mvt"] + OSM_FEATURES]
    assert len(out) == len(q), "OSM build changed the row count"
    dest = Path(f"data/processed/{side}_osm.parquet")
    out.to_parquet(dest, index=False)
    cov = out["osm_stand_rwy_dist_m"].notna().mean()
    print(f"{side:<8} {len(out):>9,} rows in {time.time()-t0:5.1f}s  "
          f"distance coverage {cov:6.1%}  -> {dest}")
