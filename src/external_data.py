"""External datasets: hourly weather (Open-Meteo) and public holidays (`holidays`).

Both sources were license-vetted against the challenge's eligibility rule that additional
datasets be "openly available under an open source license" — see SKILLS.md §1.

  * **Weather** — Open-Meteo. Data is **CC BY 4.0**, so it must be attributed (see
    :data:`WEATHER_ATTRIBUTION`); the server code is AGPLv3 but we only consume the API.

    We use the *historical forecast* endpoint rather than the ERA5 `archive` endpoint for
    one decisive reason: **the ERA5 archive does not expose `visibility`** (it returns unit
    "undefined"), and low-visibility procedures are one of the strongest known drivers of
    taxi-out time. The historical-forecast endpoint archives the high-resolution model runs
    and does carry visibility. It is also arguably the more honest choice operationally: it
    reflects what was forecast at the time rather than a post-hoc reanalysis.

  * **Holidays** — the `holidays` package (MIT), evaluated per airport with the correct
    country *and* subdivision (see :mod:`src.airports`).

Both are joined on the **takeoff time** (`MVT_TIME_UTC_mvt`), never on the off-block time:
`BLOCK_TIME_UTC_mvt` is 100% blank in `ranking.parquet`, whereas `MVT_TIME_UTC_mvt` is
fully populated there. Joining on takeoff time is therefore the only variant that works
identically in training and at prediction time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from .airports import AIRPORTS

WEATHER_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

WEATHER_ATTRIBUTION = (
    "Weather data by Open-Meteo.com (https://open-meteo.com), licensed CC BY 4.0 "
    "(https://creativecommons.org/licenses/by/4.0/)."
)

#: Hourly variables requested from Open-Meteo, chosen for plausible taxi-out relevance.
WEATHER_VARS = [
    "temperature_2m",        # de-icing regime when near/below freezing
    "dew_point_2m",          # with temperature -> spread, a fog/LVP predictor
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",              # snow clearing / de-icing -> long taxi
    "snow_depth",
    "weather_code",          # WMO code: categorical fog/snow/thunderstorm regimes
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "visibility",            # THE low-visibility-procedures driver (archive API lacks this)
    "wind_speed_10m",        # drives runway configuration changes
    "wind_direction_10m",
    "wind_gusts_10m",
    "cape",                  # convective potential -> summer thunderstorm ground stops
]

DEFAULT_WEATHER_PATH = Path("data/external/weather_hourly.parquet")
DEFAULT_HOLIDAYS_PATH = Path("data/external/holidays.parquet")


# --------------------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------------------
def fetch_weather(
    start_date: str,
    end_date: str,
    airports: pd.DataFrame | None = None,
    out_path: Path | str = DEFAULT_WEATHER_PATH,
    pause_s: float = 1.5,
    max_retries: int = 4,
) -> pd.DataFrame:
    """Download hourly weather for each airport and cache it as parquet.

    One request per airport covering the whole date range. Returns the long-format frame
    with columns ``icao``, ``time_utc`` and one column per entry in :data:`WEATHER_VARS`.
    """
    airports = AIRPORTS if airports is None else airports
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for row in airports.itertuples(index=False):
        params = {
            "latitude": row.lat,
            "longitude": row.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(WEATHER_VARS),
            "timezone": "UTC",
            "wind_speed_unit": "ms",  # m/s reads more naturally for aviation than km/h
        }
        payload = _get_json(WEATHER_URL, params, max_retries=max_retries)
        hourly = payload["hourly"]

        df = pd.DataFrame(hourly)
        df["time_utc"] = pd.to_datetime(df.pop("time"), utc=True)
        df.insert(0, "icao", row.icao)
        # keep the grid point actually used, so any spatial mismatch is auditable later
        df["grid_lat"] = payload["latitude"]
        df["grid_lon"] = payload["longitude"]
        df["grid_elevation_m"] = payload["elevation"]
        frames.append(df)
        print(f"  {row.icao}: {len(df):,} hourly rows "
              f"({df['time_utc'].min()} -> {df['time_utc'].max()})")
        time.sleep(pause_s)

    weather = pd.concat(frames, ignore_index=True)
    weather = weather.sort_values(["icao", "time_utc"]).reset_index(drop=True)
    weather.to_parquet(out_path, index=False)

    meta = {
        "source": "Open-Meteo historical forecast API",
        "endpoint": WEATHER_URL,
        "license": "CC BY 4.0",
        "attribution": WEATHER_ATTRIBUTION,
        "retrieved_utc": pd.Timestamp.utcnow().isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "variables": WEATHER_VARS,
        "airports": airports["icao"].tolist(),
        "rows": len(weather),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return weather


def _get_json(url: str, params: dict, max_retries: int = 4) -> dict:
    """GET with exponential backoff — Open-Meteo rate-limits rather than failing hard."""
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=120)
            if resp.status_code == 429:
                raise RuntimeError("rate limited (429)")
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last_error = exc
            if attempt < max_retries - 1:
                print(f"    retry {attempt + 1}/{max_retries - 1} after {exc!r}; sleeping {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def load_weather(path: Path | str = DEFAULT_WEATHER_PATH) -> pd.DataFrame:
    """Load the cached hourly weather frame."""
    return pd.read_parquet(path)


def attach_weather(
    df: pd.DataFrame,
    weather: pd.DataFrame,
    time_col: str = "MVT_TIME_UTC_mvt",
    airport_col: str = "ADEP_mvt",
    prefix: str = "wx_",
) -> pd.DataFrame:
    """Left-join hourly weather onto movements on (airport, hour of ``time_col``).

    ``time_col`` defaults to the **takeoff** time because that is the timestamp available
    in both the training files and ``ranking.parquet``.
    """
    out = df.copy()
    out["_wx_hour"] = out[time_col].dt.floor("h")

    wx = weather.rename(columns={c: f"{prefix}{c}" for c in weather.columns
                                 if c not in ("icao", "time_utc")})
    wx = wx.rename(columns={"icao": "_wx_icao", "time_utc": "_wx_hour"})

    out = out.merge(
        wx,
        left_on=[airport_col, "_wx_hour"],
        right_on=["_wx_icao", "_wx_hour"],
        how="left",
    )
    return out.drop(columns=["_wx_icao", "_wx_hour"])


# --------------------------------------------------------------------------------------
# Public holidays
# --------------------------------------------------------------------------------------
def build_holidays(
    years: list[int],
    airports: pd.DataFrame | None = None,
    out_path: Path | str = DEFAULT_HOLIDAYS_PATH,
) -> pd.DataFrame:
    """Build a per-airport daily holiday calendar in the airport's *local* time.

    Produces one row per (airport, local date) that is a public holiday, plus neighbour
    flags — the day before/after a holiday often has a more distinctive traffic mix than
    the holiday itself.
    """
    import holidays as holidays_pkg

    airports = AIRPORTS if airports is None else airports
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for row in airports.itertuples(index=False):
        # pandas stores the absent subdivisions as NaN, but `holidays` wants None
        subdiv = row.holiday_subdiv if isinstance(row.holiday_subdiv, str) else None
        cal = holidays_pkg.country_holidays(
            row.holiday_country, subdiv=subdiv, years=years
        )
        for date, name in sorted(cal.items()):
            records.append(
                {
                    "icao": row.icao,
                    "local_date": pd.Timestamp(date).date(),
                    "holiday_name": name,
                    "holiday_country": row.holiday_country,
                    "holiday_subdiv": subdiv,
                }
            )

    hol = pd.DataFrame.from_records(records)
    hol.to_parquet(out_path, index=False)

    meta = {
        "source": "python `holidays` package (vacanza/holidays)",
        "license": "MIT",
        "version": holidays_pkg.__version__,
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "years": years,
        "airports": airports["icao"].tolist(),
        "rows": len(hol),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return hol


def load_holidays(path: Path | str = DEFAULT_HOLIDAYS_PATH) -> pd.DataFrame:
    """Load the cached holiday calendar."""
    return pd.read_parquet(path)


def attach_holidays(
    df: pd.DataFrame,
    hol: pd.DataFrame,
    time_col: str = "MVT_TIME_UTC_mvt",
    airport_col: str = "ADEP_mvt",
) -> pd.DataFrame:
    """Attach holiday flags, converting the UTC takeoff time to the airport's local date.

    Adds ``is_holiday``, ``holiday_name``, ``is_day_before_holiday``,
    ``is_day_after_holiday`` and ``local_date`` / ``local_hour`` / ``local_dow``.
    """
    from .airports import TZ_BY_ICAO

    out = df.copy()

    # Local civil time per airport: convert per timezone group (vectorised within a group).
    local = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    for icao, tz in TZ_BY_ICAO.items():
        mask = out[airport_col] == icao
        if mask.any():
            local.loc[mask] = out.loc[mask, time_col].dt.tz_convert(tz).dt.tz_localize(None)
    out["local_time"] = local
    out["local_date"] = out["local_time"].dt.date
    out["local_hour"] = out["local_time"].dt.hour
    out["local_dow"] = out["local_time"].dt.dayofweek

    hol_keys = hol[["icao", "local_date", "holiday_name"]].drop_duplicates(
        subset=["icao", "local_date"]
    )

    out = out.merge(
        hol_keys.rename(columns={"icao": "_h_icao"}),
        left_on=[airport_col, "local_date"],
        right_on=["_h_icao", "local_date"],
        how="left",
    )
    out["is_holiday"] = out["holiday_name"].notna()

    # neighbour-day flags, built from the same calendar shifted by +/- 1 day
    for offset, col in ((1, "is_day_before_holiday"), (-1, "is_day_after_holiday")):
        shifted = hol_keys.copy()
        shifted["local_date"] = pd.to_datetime(shifted["local_date"]) - pd.Timedelta(days=offset)
        shifted["local_date"] = shifted["local_date"].dt.date
        shifted = shifted[["icao", "local_date"]].drop_duplicates().assign(**{col: True})
        out = out.merge(
            shifted.rename(columns={"icao": "_n_icao"}),
            left_on=[airport_col, "local_date"],
            right_on=["_n_icao", "local_date"],
            how="left",
        )
        out[col] = out[col].fillna(False)
        out = out.drop(columns=["_n_icao"])

    return out.drop(columns=["_h_icao"])
