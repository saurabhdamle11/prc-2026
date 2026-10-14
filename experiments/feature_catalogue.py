#!/usr/bin/env python
"""Complete inventory of the model's inputs, with importances, coverage and provenance.

Writes docs/features.md. Uses the shipped all-12-month booster (`lgbm_full.txt`), so the
importances are the ones that actually produced the v3 submission rather than a re-fit.
"""
from __future__ import annotations
import sys, re; from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import model_features
from src.cleaning import trainable_mask

train = pd.read_parquet("data/processed/train_features.parquet")
train = train.merge(pd.read_parquet("data/processed/train_surface.parquet"),
                    on="MVT_ID_mvt", how="left", validate="one_to_one")
rank = pd.read_parquet("data/processed/ranking_features.parquet")
rank = rank.merge(pd.read_parquet("data/processed/ranking_surface.parquet"),
                  on="MVT_ID_mvt", how="left", validate="one_to_one")
nums, cats = model_features(train)
feats = nums + cats

b = lgb.Booster(model_file="data/processed/lgbm_full.txt")
names = b.feature_name()
gain = pd.Series(b.feature_importance("gain"), index=names)
split = pd.Series(b.feature_importance("split"), index=names)
gain_pct = 100 * gain / gain.sum()

#: Which engineering step produced each column, keyed by prefix or exact name.
def family(c: str) -> str:
    if c.startswith("cong_"): return "congestion (counts, +/-W min)"
    if c.startswith("queue_"): return "queue (aircraft on the surface)"
    if c.startswith("sec_since_prev") or c.startswith("sec_to_next"): return "runway sequence"
    if c.startswith("surf_"): return "surface state (rolling realised durations)"
    if c.startswith("proxy_taxi") or c in {"offblock_delay_sec", "takeoff_vs_sched_sec"}: return "off-block proxies"
    if c.startswith("offblock_") or c == "proxy_vs_plan": return "off-block corroboration"
    if c.startswith("wx_") or c.startswith("wind_") or c.startswith("gust_"): return "weather"
    if c.startswith("dq_"): return "data-quality flags"
    if c.startswith("stand_"): return "stand / apron"
    if c in {"local_minute_of_day","local_month","local_doy","is_weekend"} or re.match(r"^(tod|dow|doy)_(sin|cos)$", c): return "time (local civil)"
    if c.startswith("is_holiday") or c.startswith("holiday") or c.startswith("days_"): return "holidays"
    if c in {"planned_enroute_sec","is_longhaul","dest_region","dest_country","is_domestic_region","ADES_mvt"}: return "route / destination"
    return "raw movement & flight-plan fields"

rows = []
for c in feats:
    src = train[c]
    rows.append({
        "feature": c,
        "family": family(c),
        "kind": "categorical" if c in cats else ("bool" if pd.api.types.is_bool_dtype(src) else "numeric"),
        "gain_%": round(float(gain_pct.get(c, 0.0)), 3),
        "splits": int(split.get(c, 0)),
        "train_present_%": round(100 * float(src.notna().mean()), 1),
        "rank_present_%": round(100 * float(rank[c].notna().mean()), 1) if c in rank.columns else np.nan,
        "n_unique": int(src.nunique(dropna=True)) if c in cats else np.nan,
    })
cat = pd.DataFrame(rows).sort_values("gain_%", ascending=False, ignore_index=True)
cat.to_csv("data/processed/feature_catalogue.csv", index=False)

print(f"{len(feats)} model inputs\n")
print("=== by family ===")
fam = (cat.groupby("family").agg(n=("feature","size"), gain=("gain_%","sum"),
                                 splits=("splits","sum")).sort_values("gain", ascending=False))
fam["gain"] = fam["gain"].round(1)
print(fam.to_string())
print("\n=== top 25 features ===")
print(cat.head(25).to_string(index=False))
print("\n=== features carrying essentially nothing (gain < 0.05%) ===")
dead = cat[cat["gain_%"] < 0.05]
print(f"{len(dead)} of {len(feats)}")
print(dead[["feature","family","gain_%","splits"]].to_string(index=False))
