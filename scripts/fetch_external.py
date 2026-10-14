#!/usr/bin/env python
"""One-shot fetch of the two vetted external datasets into `data/external/`.

Run from the repository root:

    python scripts/fetch_external.py

Both outputs are cached as parquet with a `.meta.json` sidecar recording the source,
license and retrieval time, so the provenance needed for the challenge's reproducibility
requirement travels with the data.

Date range covers the full training year (2025) and both ranking months (Jan and Jul
2026), padded by a few days at each end so lagged/rolling weather features and local-time
conversions never run off the edge of the table.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.external_data import build_holidays, fetch_weather  # noqa: E402

START_DATE = "2024-12-28"
END_DATE = "2026-08-03"
YEARS = [2024, 2025, 2026]


def main() -> None:
    print(f"Fetching hourly weather {START_DATE} -> {END_DATE} (Open-Meteo, CC BY 4.0)")
    weather = fetch_weather(START_DATE, END_DATE)
    print(f"-> data/external/weather_hourly.parquet: {weather.shape}")

    print(f"\nBuilding public-holiday calendar for {YEARS} (holidays pkg, MIT)")
    hol = build_holidays(YEARS)
    print(f"-> data/external/holidays.parquet: {hol.shape}")
    print(hol.groupby("icao").size().to_string())


if __name__ == "__main__":
    main()
