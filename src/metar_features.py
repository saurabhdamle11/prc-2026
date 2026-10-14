"""METAR-derived convective and disruption features, from IEM's public-domain archive.

Open-Meteo already supplies a gridded hourly `weather_code` and `cape`. A METAR is different in
kind: it is what the tower actually observed and reported, half-hourly plus SPECIs whenever
conditions change, so it records thunderstorms *at the aerodrome* (``TS``), in the vicinity
(``VCTS``), convective cloud (``CB``/``TCU``) and the trend forecast issued with the report
(``TEMPO TS``) — the things that trigger flow measures and ground stops in summer.

**Latency.** Every feature uses only reports issued at or before the flight's takeoff time
``MVT_TIME_UTC_mvt``, which is 100% present in the ranking set (`docs/findings.md` finding 1). A
feature is taken from the latest report at or before takeoff via a backward ``merge_asof``,
so its staleness is at most one reporting interval (~30 min), identically in both files.

Source licence: public domain (IEM disclaimer); provenance in
``data/external/metar_raw.meta.json``.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

#: Everything from the first of these onward is trend/remarks, not the observation itself.
_TREND = re.compile(r"\s(NOSIG|BECMG|TEMPO|RMK)\b")
_TS = re.compile(r"(?:^|\s)[-+]?TS[A-Z]*(?=\s|$)")
_VCTS = re.compile(r"(?:^|\s)VCTS(?=\s|$)")
_RETS = re.compile(r"(?:^|\s)RE[A-Z]*TS[A-Z]*(?=\s|$)")
_CB = re.compile(r"\d{3}CB(?=\s|$|/)")
_TCU = re.compile(r"\d{3}TCU(?=\s|$|/)")
_HEAVY = re.compile(r"(?:^|\s)\+(?:TS|SH)?(?:RA|GR|GS|SN)[A-Z]*(?=\s|$)")
_HAIL = re.compile(r"(?:^|\s)[-+]?(?:TS|SH)?[A-Z]*(?:GR|GS)(?=\s|$)")
_GUST = re.compile(r"\s(?:\d{3}|VRB)\d{2,3}G(\d{2,3})(KT|MPS)\b")
_VIS = re.compile(r"\s(\d{4})(?:NDV)?\s")
_RVR = re.compile(r"\sR\d{2}[LRC]?/[PM]?(\d{4})")

METAR_FEATURES = [
    "mt_ts_now", "mt_vcts_now", "mt_cb_now", "mt_tcu_now", "mt_heavy_now", "mt_hail_now",
    "mt_tempo_ts_now", "mt_rets_now", "mt_conv_now",
    "mt_conv_2h", "mt_conv_count_3h", "mt_ts_count_6h", "mt_min_since_conv",
    "mt_reports_1h", "mt_vis_m", "mt_rvr_min_m", "mt_gust_kt",
]


def parse_reports(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per report with boolean/numeric observation fields."""
    m = raw.copy()
    text = m["metar"].fillna("").astype(str)
    split = text.str.split(_TREND, n=1, regex=True)
    obs = split.str[0].fillna("")
    trend = text.str.extract(r"\s((?:BECMG|TEMPO)\b.*?)(?:\sRMK\b|$)", expand=False).fillna("")
    m["ts"] = obs.str.contains(_TS)
    m["vcts"] = obs.str.contains(_VCTS)
    m["rets"] = obs.str.contains(_RETS)
    m["cb"] = obs.str.contains(_CB)
    m["tcu"] = obs.str.contains(_TCU)
    m["heavy"] = obs.str.contains(_HEAVY)
    m["hail"] = obs.str.contains(_HAIL)
    m["tempo_ts"] = trend.str.contains(r"\bTS")
    m["conv"] = m["ts"] | m["vcts"] | m["cb"]
    g = obs.str.extract(_GUST)
    m["gust_kt"] = pd.to_numeric(g[0], errors="coerce") * np.where(g[1] == "MPS", 1.944, 1.0)
    vis = pd.to_numeric(obs.str.extract(_VIS)[0], errors="coerce")
    m["vis_m"] = vis.where(~obs.str.contains(r"\sCAVOK\b"), 9999.0)
    rvr = obs.str.findall(_RVR).map(lambda xs: min(map(int, xs)) if xs else np.nan)
    m["rvr_min_m"] = rvr.astype("float64")
    return m.drop(columns=["metar"])


def _per_station(r: pd.DataFrame) -> pd.DataFrame:
    """Rolling/elapsed features evaluated at each report time, backward-looking only."""
    r = r.sort_values("valid_utc").set_index("valid_utc")
    out = pd.DataFrame(index=r.index)
    for src, dst in [("ts", "mt_ts_now"), ("vcts", "mt_vcts_now"), ("cb", "mt_cb_now"),
                     ("tcu", "mt_tcu_now"), ("heavy", "mt_heavy_now"), ("hail", "mt_hail_now"),
                     ("tempo_ts", "mt_tempo_ts_now"), ("rets", "mt_rets_now"),
                     ("conv", "mt_conv_now")]:
        out[dst] = r[src].astype("int8")
    conv = r["conv"].astype("float64")
    out["mt_conv_2h"] = (conv.rolling("2h").max() > 0).astype("int8")
    out["mt_conv_count_3h"] = conv.rolling("3h").sum()
    out["mt_ts_count_6h"] = r["ts"].astype("float64").rolling("6h").sum()
    out["mt_reports_1h"] = conv.rolling("1h").count()
    last_conv = pd.Series(np.where(r["conv"], r.index.values, np.datetime64("NaT")),
                          index=r.index).ffill()
    out["_last_conv"] = last_conv.values
    out["mt_vis_m"] = r["vis_m"].values
    out["mt_rvr_min_m"] = r["rvr_min_m"].values
    out["mt_gust_kt"] = r["gust_kt"].values
    return out.reset_index()


def build_metar_features(flights: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Join report-time features onto flights by (ADEP_mvt, latest report <= takeoff).

    ``flights`` needs ``MVT_ID_mvt``, ``ADEP_mvt`` and ``MVT_TIME_UTC_mvt``. Returns one row per
    flight keyed by ``MVT_ID_mvt`` with the columns in :data:`METAR_FEATURES`.
    """
    rep = parse_reports(raw)
    parts = []
    for st, r in rep.groupby("station", sort=False):
        f = _per_station(r)
        f["station"] = st
        parts.append(f)
    rf = pd.concat(parts, ignore_index=True).sort_values("valid_utc")
    rf["valid_utc"] = rf["valid_utc"].astype("datetime64[us]")

    fl = flights[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]].copy()
    fl["t"] = fl["MVT_TIME_UTC_mvt"].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[us]")
    fl["ADEP_mvt"] = fl["ADEP_mvt"].astype("string")
    rf["station"] = rf["station"].astype("string")
    j = pd.merge_asof(fl.sort_values("t"), rf, left_on="t", right_on="valid_utc",
                      left_by="ADEP_mvt", right_by="station", direction="backward",
                      tolerance=pd.Timedelta("3h"))
    since = (j["t"] - pd.to_datetime(j["_last_conv"])).dt.total_seconds() / 60
    j["mt_min_since_conv"] = since.clip(upper=1440).fillna(1440)
    return j[["MVT_ID_mvt"] + METAR_FEATURES].reset_index(drop=True)
