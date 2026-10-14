#!/usr/bin/env python
"""Build the runway-configuration and stand-rotation features for both sides.

    python scripts/build_config_features.py

Writes ``data/processed/{train,ranking}_config.parquet``, keyed by ``MVT_ID_mvt`` and joined
one-to-one onto the feature tables. Worth -1.36 s full / -1.86 s trimmed on the mirrored
protocol; see `SKILLS.md` §11 and `src/config_features.py` for why.

The traffic universe is the *unfiltered* movement frame from each side — training rows come
from the twelve monthly files, ranking rows from ``ranking.parquet`` itself, which is a
complete movement record. Both sides therefore observe surrounding traffic at the same rate,
which is what makes these features transfer.
"""

from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_features import UNIVERSE_COLS, build_config_features  # noqa: E402
from src.cleaning import ranking_path  # noqa: E402

Q_COLS = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt",
          "AOBT_3_flt", "SCHED_TIME_UTC_mvt"]


def universe(side: str) -> pd.DataFrame:
    if side == "train":
        files = sorted(glob.glob("Dataset/training_*.parquet"))
        if not files:
            raise SystemExit("no Dataset/training_*.parquet files found")
        return pd.concat([pd.read_parquet(f, columns=UNIVERSE_COLS) for f in files],
                         ignore_index=True)
    return pd.read_parquet(ranking_path(), columns=UNIVERSE_COLS)


def main() -> None:
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    for side in ("train", "ranking"):
        src = out_dir / f"{side}_features.parquet"
        if not src.exists():
            raise SystemExit(f"{src} not found — run scripts/build_features.py first")
        t0 = time.time()
        q = pd.read_parquet(src, columns=Q_COLS)
        out = build_config_features(q, universe(side))
        assert len(out) == len(q), "config build changed the row count"
        assert out["MVT_ID_mvt"].is_unique, "duplicate MVT_ID_mvt in config output"
        dest = out_dir / f"{side}_config.parquet"
        out.to_parquet(dest, index=False)
        print(f"{side:<8} {len(out):>9,} rows in {time.time()-t0:5.0f}s  "
              f"depcfg levels {out.depcfg_cfg_id.nunique():>4}  "
              f"arrcfg levels {out.arrcfg_cfg_id.nunique():>4}  "
              f"turnaround matched {out.stand_turnaround_sec.notna().mean():.1%}  -> {dest}")


if __name__ == "__main__":
    main()
