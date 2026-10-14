#!/usr/bin/env python
"""One-shot fetch of raw METAR/SPECI reports for the ten departure airports into `data/external/`.

Run from the repository root:

    python scripts/fetch_metar.py

Source: the Iowa Environmental Mesonet (IEM) global ASOS/METAR archive, Iowa State University.
IEM's disclaimer (https://mesonet.agron.iastate.edu/disclaimer.php) states: "The materials
found on this website are in the public domain and may be used freely by anyone for any lawful
purpose." Attribution is appreciated but not required; we give it anyway.

Output: `data/external/metar_raw.parquet` (station, valid_utc, metar) plus a `.meta.json`
provenance sidecar. Same padded date range as `fetch_external.py`, so rolling windows never run
off the edge of the table. One request per airport per calendar year, politely spaced.
"""
from __future__ import annotations

import io, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.airports import DEPARTURE_AIRPORTS  # noqa: E402

ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
START, END = "2024-12-28", "2026-08-03"
OUT = Path("data/external/metar_raw.parquet")


def fetch(station: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    q = {"station": station, "data": "metar", "tz": "Etc/UTC", "format": "onlycomma",
         "latlon": "no", "missing": "M", "trace": "T", "direct": "no",
         "report_type": ["3", "4"],  # 3 = routine METAR, 4 = SPECI
         "year1": start.year, "month1": start.month, "day1": start.day,
         "year2": end.year, "month2": end.month, "day2": end.day}
    for attempt in range(5):
        r = requests.get(ENDPOINT, params=q, timeout=300)
        if r.ok and r.text.startswith("station"):
            return pd.read_csv(io.StringIO(r.text))
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{station} {start:%Y-%m-%d}: HTTP {r.status_code} {r.text[:200]}")


def main() -> None:
    start, end = pd.Timestamp(START), pd.Timestamp(END)
    edges = [start] + [pd.Timestamp(f"{y}-01-01") for y in range(start.year + 1, end.year + 1)] + [end]
    parts = []
    for apt in DEPARTURE_AIRPORTS:
        for a, b in zip(edges[:-1], edges[1:]):
            df = fetch(apt, a, b)
            print(f"{apt} {a:%Y-%m-%d} -> {b:%Y-%m-%d}: {len(df):,} reports", flush=True)
            parts.append(df)
            time.sleep(2)
    m = pd.concat(parts, ignore_index=True)
    m["valid_utc"] = pd.to_datetime(m["valid"], utc=True).dt.tz_localize(None)
    m = (m[["station", "valid_utc", "metar"]].drop_duplicates(["station", "valid_utc", "metar"])
         .sort_values(["station", "valid_utc"], ignore_index=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(OUT, index=False)
    meta = {
        "source": "Iowa Environmental Mesonet (IEM), Iowa State University — global ASOS/METAR archive",
        "endpoint": ENDPOINT,
        "license": "Public domain",
        "license_statement": ("The materials found on this website are in the public domain and "
                              "may be used freely by anyone for any lawful purpose."),
        "license_url": "https://mesonet.agron.iastate.edu/disclaimer.php",
        "attribution": "METAR data courtesy of the Iowa Environmental Mesonet, Iowa State University.",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": START, "end_date": END, "report_types": ["METAR", "SPECI"],
        "airports": list(DEPARTURE_AIRPORTS),
        "rows": int(len(m)),
    }
    OUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"-> {OUT}: {len(m):,} reports")


if __name__ == "__main__":
    main()
