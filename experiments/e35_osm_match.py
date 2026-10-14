#!/usr/bin/env python
"""E35: do OSM stand and runway references actually join to the challenge data?

The kill-switch for the OSM taxi-distance feature, run before building anything. Geometry is
only useful if it can be attached to our rows, and that needs two joins to land:

  * `parking_position` `ref` (volunteer-maintained, format varies by airport) -> `STAND_mvt`
  * `runway` `ref` such as "07L/25R" -> `RUNWAY_mvt` such as "25R", picking the correct
    *threshold*: a departure on 25R starts at the 25R end, i.e. the endpoint whose bearing
    toward the other end is ~250 degrees.

Reports match rates weighted by how many departures each stand actually carries — a stand seen
twice a year matching is worth nothing; the busy ones are what count.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.airports import AIRPORTS

OSM = Path("data/external/osm")


def centroid(el: dict) -> tuple[float, float] | None:
    if el["type"] == "node":
        return el["lat"], el["lon"]
    g = el.get("geometry") or []
    if not g:
        return None
    return float(np.mean([p["lat"] for p in g])), float(np.mean([p["lon"] for p in g]))


def load(icao: str) -> tuple[dict, dict]:
    """(stand ref -> centroid, runway designator -> threshold point)."""
    d = json.loads((OSM / f"{icao}.json").read_text())
    stands, runways = {}, {}
    for el in d.get("elements", []):
        t = el.get("tags", {})
        if t.get("aeroway") == "parking_position":
            ref = (t.get("ref") or t.get("name") or "").strip()
            c = centroid(el)
            if ref and c:
                stands[ref] = c
        elif t.get("aeroway") == "runway":
            g = el.get("geometry") or []
            ref = (t.get("ref") or "").strip()
            if len(g) < 2 or not ref:
                continue
            a, b = (g[0]["lat"], g[0]["lon"]), (g[-1]["lat"], g[-1]["lon"])
            for desig in [x.strip() for x in ref.split("/")]:
                num = "".join(ch for ch in desig if ch.isdigit())
                if not num:
                    continue
                want = float(num) * 10.0
                # the threshold for designator D is the end you depart FROM: the bearing from
                # it toward the far end must be ~D*10 degrees
                for start, end in ((a, b), (b, a)):
                    dy = end[0] - start[0]
                    dx = (end[1] - start[1]) * np.cos(np.radians(start[0]))
                    brg = (np.degrees(np.arctan2(dx, dy))) % 360
                    if min(abs(brg - want), 360 - abs(brg - want)) <= 30:
                        runways[desig] = start
                        break
    return stands, runways


tr = pd.read_parquet("Dataset/training_2025-01-01_2025-02-01.parquet",
                     columns=["PHASE_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])
dep = tr[tr.PHASE_mvt == "DEP"]
print(f"{'airport':<8} {'OSM stands':>11} {'OSM rwys':>9} {'stand match':>26} {'runway match':>26}")
tot_w = tot_m = 0
for r in AIRPORTS[AIRPORTS["has_departures"]].itertuples():
    f = OSM / f"{r.icao}.json"
    if not f.exists():
        print(f"{r.icao:<8} {'(not fetched)':>11}")
        continue
    stands, runways = load(r.icao)
    d = dep[dep.ADEP_mvt == r.icao]
    sc = d.STAND_mvt.astype("string").value_counts()
    rc = d.RUNWAY_mvt.astype("string").value_counts()
    s_hit = sum(v for k, v in sc.items() if k in stands)
    r_hit = sum(v for k, v in rc.items() if k in runways)
    tot_w += sc.sum(); tot_m += s_hit
    print(f"{r.icao:<8} {len(stands):>11,} {len(runways):>9} "
          f"{s_hit:>10,}/{sc.sum():<9,} ({s_hit/max(sc.sum(),1):>5.1%}) "
          f"{r_hit:>10,}/{rc.sum():<9,} ({r_hit/max(rc.sum(),1):>5.1%})")
    if len(stands) and s_hit / max(sc.sum(), 1) < 0.2:
        print(f"         OSM refs : {sorted(stands)[:8]}")
        print(f"         our stands: {sc.index[:8].tolist()}")
print(f"\nDEPARTURE-WEIGHTED STAND MATCH: {tot_m:,}/{tot_w:,} = {tot_m/max(tot_w,1):.1%}")
