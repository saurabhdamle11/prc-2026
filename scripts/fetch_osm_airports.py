#!/usr/bin/env python
"""Fetch airport surface geometry from OpenStreetMap for the ten departure airports.

    python scripts/fetch_osm_airports.py

Taxi time is, physically, distance / speed plus queueing. The model currently infers distance
from categorical `stand_area` (176 levels), `stand_full` (2,723) and `RUNWAY_mvt` (53), which
works where a stand-runway pair is common and generalises poorly where it is rare. Real
geometry gives a *numeric* distance that orders every pair sensibly, including ones seen a
handful of times.

It also covers **all ten airports**, unlike the ADS-B route which had no surface coverage at
LIRF, LTFM, LFPG or LEMD (`docs/adsb_offblock_plan.md`).

Source: OpenStreetMap via the Overpass API, **ODbL 1.0** — open, documented, and satisfying the
contest's supplementary-data rule.

Attribution required in any published output:
  "Airport geometry (c) OpenStreetMap contributors, licensed under ODbL 1.0."
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.airports import AIRPORTS  # noqa: E402

#: The main endpoint 504s intermittently under load; the mirrors serve the same data, so
#: rotating through them is more reliable than retrying one harder.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

#: Overpass sits behind an Apache that returns 406 Not Acceptable to requests carrying the
#: default python-requests User-Agent. Identifying the client is also basic courtesy to a
#: free community service.
HEADERS = {"User-Agent": "prc-data-challenge-2026/1.0 (research; taxi-out prediction)"}
BOX_DEG = 0.10
OUT = Path("data/external/osm")

QUERY = """
[out:json][timeout:180];
(
  way["aeroway"="runway"]({s},{w},{n},{e});
  way["aeroway"="taxiway"]({s},{w},{n},{e});
  node["aeroway"="parking_position"]({s},{w},{n},{e});
  way["aeroway"="parking_position"]({s},{w},{n},{e});
  way["aeroway"="apron"]({s},{w},{n},{e});
);
out geom;
"""


def fetch(icao: str, lat: float, lon: float) -> dict:
    q = QUERY.format(s=lat - BOX_DEG, w=lon - BOX_DEG, n=lat + BOX_DEG, e=lon + BOX_DEG)
    for attempt in range(8):
        url = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        try:
            r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=300)
        except requests.RequestException as exc:                     # noqa: BLE001
            print(f"    {url.split('/')[2]}: {type(exc).__name__}", flush=True)
            time.sleep(10)
            continue
        if r.status_code == 200:
            return r.json()
        wait = 10 * (attempt // len(OVERPASS_ENDPOINTS) + 1)
        print(f"    {url.split('/')[2]}: HTTP {r.status_code}, next mirror in {wait}s", flush=True)
        time.sleep(wait)
    raise SystemExit(f"{icao}: every Overpass mirror failed")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for r in AIRPORTS[AIRPORTS["has_departures"]].itertuples():
        dest = OUT / f"{r.icao}.json"
        if dest.exists():
            print(f"{r.icao}  cached")
            continue
        t0 = time.time()
        data = fetch(r.icao, r.lat, r.lon)
        dest.write_text(json.dumps(data))
        els = data.get("elements", [])
        kinds = pd.Series([e.get("tags", {}).get("aeroway") for e in els]).value_counts()
        print(f"{r.icao}  {len(els):>6,} elements in {time.time()-t0:5.1f}s  "
              f"{dict(kinds)}  -> {dest} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)
        time.sleep(3)          # be polite to a free community service


if __name__ == "__main__":
    main()
