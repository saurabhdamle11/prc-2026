#!/usr/bin/env python
"""Extract airport-surface ADS-B tracks from the adsb.lol open historical archive.

    python scripts/fetch_adsb_ground.py --date 2025-01-15        # one day (the pilot)
    python scripts/fetch_adsb_ground.py --date 2026-01-01 --days 31

**Why this source and not OpenSky.** The contest requires every supplementary dataset to be
openly accessible and open-licensed (challenge eligibility rules). OpenSky's historical
Trino database is free for approved researchers but is *not* open-licensed, so it cannot be
used here. **adsb.lol publishes its full historical archive under ODbL 1.0** as public GitHub
releases — no access request, no review, no gatekeeper — which satisfies the rule outright.

  https://github.com/adsblol/globe_history_2026   (and _2025)
  https://www.adsb.lol/docs/open-data/historical/

Attribution required in any published output:
  "Historical ADS-B data by adsb.lol, licensed under ODbL 1.0."

**Why streaming.** A day is 2.2-4.1 GB compressed and the four target months total ~404 GB,
against ~72 GB of free disk here. So each day is downloaded, streamed through `tarfile` without
ever being unpacked (a day holds hundreds of thousands of per-aircraft files, which no
filesystem enjoys), reduced to just the ground points inside the ten airport boxes, and the raw
archive deleted. Peak disk stays at roughly one day's archive.

**Archive layout** (verified 2026-09-26 by reading a real release). The tar root holds
``README.txt``, the two licence files, ``acas/``, ``heatmap/`` and ``traces/``. Aircraft files
live at ``./traces/<last two hex of icao24>/trace_full_<icao24>.json`` and are **gzip-compressed
despite the bare .json extension** — the magic bytes are ``\x1f\x8b\x08\x00``. Filtering on
``.json.gz`` therefore matches nothing but ``acas/acas.json.gz``. Names beginning ``~`` are
TIS-B/ADS-R targets rather than real ICAO addresses.

**The trace format** (wiedehopf/readsb). Each aircraft-day is one gzipped JSON with a base
``timestamp`` and a ``trace`` array whose entries are::

    [seconds_after_timestamp, lat, lon, altitude, gs, track, flags, vert_rate, aircraft, ...]

``altitude`` is the literal string ``"ground"`` when the aircraft is on the surface, and ``gs``
is ground speed in **knots**. That pair is exactly what a pushback detector needs, and it is
what `AOBT_3_flt` fails to resolve: see `docs/adsb_offblock_plan.md`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.airports import AIRPORTS  # noqa: E402

try:                                     # 3-5x faster than the stdlib on these files
    import orjson
    _loads = orjson.loads
except ImportError:
    _loads = json.loads

BOX_DEG = 0.12                           # ~13 km, comfortably covers any of these airfields
API = "https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}"
OUT_DIR = Path("data/external/adsb_ground")


def boxes() -> list[tuple[str, float, float, float, float]]:
    a = AIRPORTS[AIRPORTS["has_departures"]]
    return [(r.icao, r.lat - BOX_DEG, r.lat + BOX_DEG, r.lon - BOX_DEG, r.lon + BOX_DEG)
            for r in a.itertuples()]


def which_airport(lat: float, lon: float, bx: list) -> str | None:
    for icao, la0, la1, lo0, lo1 in bx:
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return icao
    return None


def release_assets(date: pd.Timestamp, instance: str) -> list[dict]:
    """Asset list for one day, newest instance first. Prefers the largest release for a day."""
    tag = f"v{date:%Y.%m.%d}-planes-readsb-{instance}"
    r = requests.get(API.format(year=date.year, tag=tag), timeout=60)
    if r.status_code != 200:
        return []
    return sorted(r.json().get("assets", []), key=lambda a: a["name"])


def download(assets: list[dict], dest: Path) -> Path:
    """Fetch and concatenate the split segments into one tar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        for a in assets:
            with requests.get(a["browser_download_url"], stream=True, timeout=600) as resp:
                resp.raise_for_status()
                shutil.copyfileobj(resp.raw, out, length=1 << 22)
    return dest


#: Set once per worker by the pool initialiser, so the box list is not pickled per task.
_BX: list = []


def _init_worker(bx: list) -> None:
    global _BX
    _BX = bx


def _parse_one(blob: bytes) -> list[tuple]:
    """Reduce one aircraft-day to its surface points inside the airport boxes.

    Runs in a worker process: the tar is read sequentially in the parent (it is a stream and
    cannot be seeked), while JSON parsing — which is the actual bottleneck — fans out.
    """
    try:
        d = _loads(gzip.decompress(blob))
    except Exception:                                    # noqa: BLE001 - corrupt member
        return []
    trace = d.get("trace") or []
    base, icao = d.get("timestamp"), d.get("icao")
    if not trace or base is None:
        return []
    flight = None
    keep = []
    for e in trace:
        if e[3] != "ground":                             # altitude field; "ground" is literal
            continue
        apt = which_airport(e[1], e[2], _BX)
        if apt is None:
            continue
        if flight is None and len(e) > 8 and isinstance(e[8], dict):
            f = e[8].get("flight")
            if f:
                flight = f.strip()
        keep.append((icao, apt, base + e[0], e[1], e[2], e[4]))
    return [(i, a, t, la, lo, gs, flight) for i, a, t, la, lo, gs in keep]


def _members(tar_path: Path):
    """Yield the raw bytes of every aircraft file, streaming (no unpack, no seek)."""
    with tarfile.open(tar_path, mode="r|") as tf:
        for member in tf:
            # gzip-compressed despite the .json name; `trace_full_` is the whole day
            if not member.isfile() or "/traces/" not in member.name:
                continue
            if not member.name.endswith(".json") or "trace_full_" not in member.name:
                continue
            fh = tf.extractfile(member)
            if fh is not None:
                yield fh.read()


def extract_ground(tar_path: Path, bx: list, workers: int) -> pd.DataFrame:
    """Stream the archive, keeping only ground positions inside the airport boxes."""
    rows: list[tuple] = []
    n_files = n_hit = 0
    if workers <= 1:
        _init_worker(bx)
        for blob in _members(tar_path):
            n_files += 1
            got = _parse_one(blob)
            if got:
                n_hit += 1
                rows.extend(got)
    else:
        with mp.Pool(workers, initializer=_init_worker, initargs=(bx,)) as pool:
            for got in pool.imap_unordered(_parse_one, _members(tar_path), chunksize=200):
                n_files += 1
                if got:
                    n_hit += 1
                    rows.extend(got)
    print(f"    {n_files:,} aircraft-files scanned, {n_hit:,} had surface points at our airports")
    return pd.DataFrame(rows, columns=["icao24", "airport", "time", "lat", "lon", "gs", "callsign"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="UTC day, YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--instance", default="prod-0", help="prod-0 | staging-0")
    ap.add_argument("--keep-raw", action="store_true", help="do not delete the archive after use")
    ap.add_argument("--work", type=Path, default=Path("data/external/_adsb_tmp"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parse processes. JSON parsing dominates, so this is what a large "
                         "instance buys; the tar read stays sequential in the parent.")
    args = ap.parse_args()

    bx = boxes()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.date)
    print(f"{len(bx)} airports, box +/-{BOX_DEG} deg, {args.workers} parse workers, "
          f"json: {'orjson' if _loads is not json.loads else 'stdlib (pip install orjson)'}\n")

    for i in range(args.days):
        day = start + pd.Timedelta(days=i)
        out = OUT_DIR / f"{day:%Y-%m-%d}.parquet"
        if out.exists():
            print(f"{day:%Y-%m-%d}  already done -> {out}")
            continue
        assets = release_assets(day, args.instance)
        if not assets:
            print(f"{day:%Y-%m-%d}  NO RELEASE for instance {args.instance} — skipped")
            continue
        gb = sum(a["size"] for a in assets) / 1e9
        print(f"{day:%Y-%m-%d}  {len(assets)} segments, {gb:.2f} GB", flush=True)

        t0 = time.time()
        tar = args.work / f"{day:%Y-%m-%d}.tar"
        download(assets, tar)
        t_dl = time.time() - t0
        print(f"    downloaded in {t_dl/60:.1f} min ({gb/max(t_dl,1)*1000:.0f} MB/s)", flush=True)

        t1 = time.time()
        df = extract_ground(tar, bx, args.workers)
        df.to_parquet(out, index=False)
        if not args.keep_raw:
            tar.unlink(missing_ok=True)
        print(f"    {len(df):,} surface points -> {out} "
              f"({out.stat().st_size/1e6:.1f} MB, parse {(time.time()-t1)/60:.1f} min)")
        if len(df):
            print("    per-airport points: "
                  + ", ".join(f"{k}={v:,}" for k, v in df.airport.value_counts().items()))
        print(f"    TOTAL {(time.time()-t0)/60:.1f} min/day -> "
              f"{(time.time()-t0)/60*124/60:.1f} h for the full 124-day plan", flush=True)


if __name__ == "__main__":
    main()
