#!/usr/bin/env python
"""Measure whether the ADS-B off-block plan is feasible, before committing to it.

    python scripts/probe_opensky.py --step 1     # how big is one hour partition?
    python scripts/probe_opensky.py --step 2     # what ground coverage do we get?

Run this the hour Trino access arrives. It answers, in order, the two questions that decide
the project — and either can kill it cheaply:

**Step 1 — scan cost.** `state_vectors_data4` is partitioned by `hour`, and `lat`/`lon` are
not partition columns, so an airport bounding box still scans the whole *global* partition.
Cost is therefore proportional to hours queried, against a hard **100 GB per-query scan
limit**. One hour tells us bytes-per-partition, and 2,928 x that tells us whether the scoped
plan (Jan+Jul 2025 and Jan+Jul 2026) is affordable at all.

**Step 2 — ground coverage.** The fraction of our departures that actually have usable ADS-B
surface tracks. This matters more than detector precision: a detector that is excellent on 70%
of flights and absent on the rest blends back toward `AOBT_3_flt` and gains little. Reported
per airport, because `jul LIRF` alone is 35.6% of the metric (`SKILLS.md` §11) — a low number
there guts the thesis regardless of the global average.

Neither step tunes any detection logic. That comes after both pass.

Setup::

    pip install pyopensky          # or: traffic
    # then configure credentials per https://mode-s.org/pyopensky/

Docs: https://openskynetwork.github.io/opensky-api/trino.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.airports import AIRPORTS  # noqa: E402

#: Half-width of the bounding box around each airport, in degrees. ~0.12 deg of latitude is
#: ~13 km, comfortably larger than any of these airfields including their outer taxiways.
BOX_DEG = 0.12

#: A quiet-ish but representative hour to size the partition with: a weekday morning peak.
PROBE_HOUR = "2025-01-15 08:00"


def airport_boxes() -> pd.DataFrame:
    """Bounding boxes for the ten departure airports.

    Coordinates come from `src/airports.py` (OurAirports, public domain). LTAI is excluded:
    it is listed by the organizers but has no DEP movements in the delivered data, so it would
    only widen the scan for nothing.
    """
    a = AIRPORTS[AIRPORTS["has_departures"]].copy()
    return pd.DataFrame({
        "icao": a["icao"].to_numpy(), "lat": a["lat"].to_numpy(), "lon": a["lon"].to_numpy(),
        "lat0": a["lat"] - BOX_DEG, "lat1": a["lat"] + BOX_DEG,
        "lon0": a["lon"] - BOX_DEG, "lon1": a["lon"] + BOX_DEG,
    }).reset_index(drop=True)


def bbox_predicate(boxes: pd.DataFrame) -> str:
    """One OR-ed lat/lon predicate covering all ten airports."""
    return " OR ".join(
        f"(lat BETWEEN {r.lat0:.4f} AND {r.lat1:.4f} AND lon BETWEEN {r.lon0:.4f} AND {r.lon1:.4f})"
        for r in boxes.itertuples())


def step1_sql(hour_unix: int, boxes: pd.DataFrame) -> str:
    return f"""
-- STEP 1: size one hour partition. Check the Trino UI for "bytes scanned" after running;
-- that number x 2,928 is the scoped plan's total scan, against a 100 GB per-query limit.
SELECT count(*) AS ground_rows,
       count(DISTINCT icao24) AS aircraft,
       count(DISTINCT callsign) AS callsigns
FROM state_vectors_data4
WHERE hour = {hour_unix}
  AND onground = true
  AND time - lastcontact <= 15
  AND ({bbox_predicate(boxes)})
""".strip()


def step2_sql(hour_start: int, hour_end: int, boxes: pd.DataFrame) -> str:
    return f"""
-- STEP 2: per-callsign ground presence over a day, to join against our DEP rows and measure
-- coverage. `velocity` is m/s. Aggregated server-side so the result is one row per movement
-- rather than millions of state vectors.
SELECT callsign,
       icao24,
       min(time) AS first_seen,
       max(time) AS last_seen,
       count(*)  AS n_points,
       min_by(lat, time) AS first_lat,
       min_by(lon, time) AS first_lon,
       sum(CASE WHEN velocity <= 0.5 THEN 1 ELSE 0 END) AS n_stationary,
       max(velocity) AS max_velocity
FROM state_vectors_data4
WHERE hour >= {hour_start} AND hour < {hour_end}
  AND onground = true
  AND time - lastcontact <= 15
  AND ({bbox_predicate(boxes)})
GROUP BY callsign, icao24
HAVING count(*) >= 10
""".strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=[1, 2], required=True)
    ap.add_argument("--hour", default=PROBE_HOUR, help="UTC hour for step 1")
    ap.add_argument("--day", default="2025-01-15", help="UTC day for step 2")
    ap.add_argument("--print-only", action="store_true",
                    help="print the SQL without running it (no credentials needed)")
    args = ap.parse_args()

    boxes = airport_boxes()
    if args.step == 1:
        h = int(pd.Timestamp(args.hour, tz="UTC").timestamp())
        sql = step1_sql(h, boxes)
    else:
        d0 = pd.Timestamp(args.day, tz="UTC")
        sql = step2_sql(int(d0.timestamp()), int((d0 + pd.Timedelta(days=1)).timestamp()), boxes)

    print(sql)
    if args.print_only:
        return

    try:
        from pyopensky.trino import Trino
    except ImportError:
        raise SystemExit(
            "\npyopensky is not installed. This script prints runnable SQL regardless "
            "(--print-only); to execute it here:\n"
            "    pip install pyopensky\n"
            "then configure credentials per https://mode-s.org/pyopensky/")

    print("\nrunning...", flush=True)
    df = Trino().query(sql)
    print(df.to_string())

    if args.step == 2:
        out = Path("data/external/adsb_probe.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"\n-> {out}")
        print("Next: join `callsign` against CALLSIGN_flt on that day's DEP rows and report "
              "per-airport coverage. Below ~80% at LIRF, reconsider the whole plan.")


if __name__ == "__main__":
    main()
