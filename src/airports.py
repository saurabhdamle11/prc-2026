"""Reference table for the challenge airports.

Coordinates and elevation are taken from OurAirports (https://ourairports.com/data/),
released into the **public domain** (GitHub mirror uses The Unlicense), so they can be
embedded here and redistributed without restriction. Values were extracted from
`airports.csv` (snapshot downloaded 2026-09-02) and are held as literals so the pipeline
has no runtime network dependency on that source.

Timezone and holiday-subdivision columns are added by us:
  * `tz` is the airport's IANA local timezone — needed because holidays, curfews and
    "local rush hour" are local-time concepts while all data timestamps are UTC.
  * `holiday_country` / `holiday_subdiv` are the codes accepted by the `holidays`
    package (MIT). Subdivisions matter: German and Spanish public holidays are
    state/region specific, so Frankfurt (Hesse) and Munich (Bavaria) do NOT share a
    calendar, and neither do Barcelona (Catalonia) and Madrid.

NOTE ON LTAI: the challenge documentation lists Antalya (LTAI) as one of 11 airports,
but the delivered data contains **no DEP movements** for it (it only ever appears as the
other endpoint of a flight). Effectively the task covers 10 airports. LTAI is kept in
this table so the discrepancy is explicit rather than silently dropped.
"""

from __future__ import annotations

import pandas as pd

# icao, iata, name, lat, lon, elevation_ft, holiday_country, holiday_subdiv, tz, has_departures
_ROWS = [
    ("EDDF", "FRA", "Frankfurt Main",        50.026706,  8.558350,  364.0, "DE", "HE",  "Europe/Berlin",    True),
    ("EDDM", "MUC", "Munich",                48.353802, 11.786100, 1487.0, "DE", "BY",  "Europe/Berlin",    True),
    ("EGLL", "LHR", "London Heathrow",       51.470748, -0.459909,   83.0, "GB", "ENG", "Europe/London",    True),
    ("EHAM", "AMS", "Amsterdam Schiphol",    52.308601,  4.763890,  -11.0, "NL", None,  "Europe/Amsterdam", True),
    ("LEBL", "BCN", "Barcelona-El Prat",     41.297100,  2.078460,   12.0, "ES", "CT",  "Europe/Madrid",    True),
    ("LEMD", "MAD", "Madrid-Barajas",        40.493407, -3.572249, 1998.0, "ES", "MD",  "Europe/Madrid",    True),
    ("LFPG", "CDG", "Paris Charles de Gaulle", 49.008960, 2.554117,  392.0, "FR", None,  "Europe/Paris",     True),
    ("LIRF", "FCO", "Rome Fiumicino",        41.804532, 12.251998,   13.0, "IT", "RM",  "Europe/Rome",      True),
    ("LSZH", "ZRH", "Zurich",                47.458056,  8.548056, 1417.0, "CH", "ZH",  "Europe/Zurich",    True),
    ("LTFM", "IST", "Istanbul",              41.274874, 28.732136,  325.0, "TR", None,  "Europe/Istanbul",  True),
    # listed by the organizers but absent from the delivered DEP movements — see module docstring
    ("LTAI", "AYT", "Antalya",               36.898701, 30.800501,  177.0, "TR", None,  "Europe/Istanbul",  False),
]

AIRPORTS = pd.DataFrame(
    _ROWS,
    columns=[
        "icao", "iata", "name", "lat", "lon", "elevation_ft",
        "holiday_country", "holiday_subdiv", "tz", "has_departures",
    ],
)

#: The 10 airports that actually have departure movements — the real prediction scope.
DEPARTURE_AIRPORTS: list[str] = AIRPORTS.loc[AIRPORTS["has_departures"], "icao"].tolist()

#: All 11 airports named by the challenge documentation.
ALL_AIRPORTS: list[str] = AIRPORTS["icao"].tolist()

TZ_BY_ICAO: dict[str, str] = dict(zip(AIRPORTS["icao"], AIRPORTS["tz"]))
