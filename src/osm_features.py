"""Taxi geometry from OpenStreetMap: how far is this stand from this runway?

Taxi time is, physically, distance / speed plus queueing. The shipped model infers distance
from categorical `stand_area` (176 levels), `stand_full` (2,723) and `RUNWAY_mvt` (53). That
works where a stand-runway pair is common and generalises poorly where it is rare — and there
are thousands of rare pairs. A *numeric* distance orders every pair correctly from geometry
alone, including ones seen a handful of times.

Unlike the abandoned ADS-B route (`docs/adsb_offblock_plan.md`), this covers **all ten
airports**. Measured join rates, weighted by departures (`experiments/e35_osm_match.py`):
**88.1% of stands overall and 100% at LIRF**, which alone carries 42% of the weighted MSE.
Runway thresholds match 100% everywhere.

Source: OpenStreetMap via Overpass, **ODbL 1.0**.
Attribution required in any published output:
  "Airport geometry (c) OpenStreetMap contributors, licensed under ODbL 1.0."

.. note::
   Distances are straight-line, not taxiway-routed. A real route is longer and the ratio
   varies by airport layout, but the *ordering* within an airport is what a tree splits on,
   and that survives. Routing over the OSM taxiway graph is the obvious refinement if this
   earns its place.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .airports import AIRPORTS

OSM_DIR = Path("data/external/osm")

#: Tolerance when deciding which end of a runway is the threshold for a given designator.
_BEARING_TOL_DEG = 30.0

EARTH_R = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres, vectorised."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _centroid(el: dict) -> tuple[float, float] | None:
    if el["type"] == "node":
        return el["lat"], el["lon"]
    g = el.get("geometry") or []
    if not g:
        return None
    return float(np.mean([p["lat"] for p in g])), float(np.mean([p["lon"] for p in g]))


def _bearing(start: tuple[float, float], end: tuple[float, float]) -> float:
    dy = end[0] - start[0]
    dx = (end[1] - start[1]) * np.cos(np.radians(start[0]))
    return float(np.degrees(np.arctan2(dx, dy)) % 360)


def load_airport_geometry(icao: str) -> tuple[dict, dict]:
    """``(stand ref -> centroid, runway designator -> departure threshold)``.

    The threshold for designator ``D`` is the end an aircraft departs *from*: the endpoint
    whose bearing toward the far end matches ``D * 10`` degrees. Taking the wrong end would
    invert the distance for every flight on that runway.
    """
    path = OSM_DIR / f"{icao}.json"
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text())
    stands: dict[str, tuple[float, float]] = {}
    runways: dict[str, tuple[float, float]] = {}
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        kind = tags.get("aeroway")
        if kind == "parking_position":
            ref = (tags.get("ref") or tags.get("name") or "").strip()
            c = _centroid(el)
            if ref and c:
                stands[ref] = c
        elif kind == "runway":
            geom = el.get("geometry") or []
            ref = (tags.get("ref") or "").strip()
            if len(geom) < 2 or not ref:
                continue
            a = (geom[0]["lat"], geom[0]["lon"])
            b = (geom[-1]["lat"], geom[-1]["lon"])
            for desig in (x.strip() for x in ref.split("/")):
                num = "".join(ch for ch in desig if ch.isdigit())
                if not num:
                    continue
                want = float(num) * 10.0
                for start, end in ((a, b), (b, a)):
                    brg = _bearing(start, end)
                    if min(abs(brg - want), 360 - abs(brg - want)) <= _BEARING_TOL_DEG:
                        runways[desig] = start
                        break
    return stands, runways


def geometry_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """All ten airports' stands and runway thresholds as joinable frames."""
    srows, rrows = [], []
    for icao in AIRPORTS.loc[AIRPORTS["has_departures"], "icao"]:
        stands, runways = load_airport_geometry(icao)
        srows += [{"ADEP_mvt": icao, "STAND_mvt": k, "s_lat": v[0], "s_lon": v[1]}
                  for k, v in stands.items()]
        rrows += [{"ADEP_mvt": icao, "RUNWAY_mvt": k, "r_lat": v[0], "r_lon": v[1]}
                  for k, v in runways.items()]
    return pd.DataFrame(srows), pd.DataFrame(rrows)


def add_osm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach taxi-geometry features. Null where the stand or runway is unmapped.

    Done with merges rather than per-row lookups: at 2.1M training rows the loop version
    took minutes, and this is the same arithmetic.
    """
    out = df.copy()
    stands, runways = geometry_tables()
    key = pd.DataFrame({
        "ADEP_mvt": out["ADEP_mvt"].astype("string"),
        "STAND_mvt": out["STAND_mvt"].astype("string"),
        "RUNWAY_mvt": out["RUNWAY_mvt"].astype("string"),
    })
    for t in (stands, runways):
        for c in t.columns:
            if t[c].dtype == object:
                t[c] = t[c].astype("string")
    key = key.merge(stands, on=["ADEP_mvt", "STAND_mvt"], how="left")
    key = key.merge(runways, on=["ADEP_mvt", "RUNWAY_mvt"], how="left")
    arp = AIRPORTS.set_index("icao")[["lat", "lon"]]
    key["a_lat"] = key["ADEP_mvt"].map(arp["lat"]).astype("float64")
    key["a_lon"] = key["ADEP_mvt"].map(arp["lon"]).astype("float64")

    out["osm_stand_rwy_dist_m"] = haversine_m(key.s_lat, key.s_lon, key.r_lat, key.r_lon)
    out["osm_stand_arp_dist_m"] = haversine_m(key.s_lat, key.s_lon, key.a_lat, key.a_lon)

    # deviation of the straight line to the threshold from the runway heading: a large value
    # means crossing the field rather than running alongside it
    num = key["RUNWAY_mvt"].str.extract(r"^(\d{1,2})", expand=False).astype("Float64") * 10.0
    dy = key.r_lat - key.s_lat
    dx = (key.r_lon - key.s_lon) * np.cos(np.radians(key.s_lat))
    brg = (np.degrees(np.arctan2(dx, dy))) % 360
    dev = (brg - num.astype("float64")).abs() % 360
    out["osm_stand_rwy_bearing_dev"] = np.minimum(dev, 360 - dev)

    # airport-free: a 2 km taxi is long at LSZH and short at LFPG
    med = out.groupby("ADEP_mvt", observed=True)["osm_stand_rwy_dist_m"].transform("median")
    with np.errstate(invalid="ignore", divide="ignore"):
        out["osm_dist_ratio"] = out["osm_stand_rwy_dist_m"] / med
    return out


OSM_FEATURES = ["osm_stand_rwy_dist_m", "osm_stand_arp_dist_m",
                "osm_stand_rwy_bearing_dev", "osm_dist_ratio"]
