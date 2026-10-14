#!/usr/bin/env python
"""E23: signed pairwise gaps between the off-block estimates.

E22 found the mechanism behind the long-taxi error, and it is not shrinkage:

    true taxi band      y    px_aobt3   px_eobt1   AOBT_3 - BLOCK (the hold)
    [   0, 1200)      833        873       1144        -67
    [1200, 1800)     1381       1207       1741       +122
    [1800, 2700)     2008       1435       2401       +478
    [2700, 3600)     2995       1199       3183      +1376
    [3600,   inf)    4312       1079       4014      +2758

`proxy_taxi_aobt3` -- the single highest-gain feature in the model -- **inverts** in the tail:
it falls from 1,435s to 1,079s while the true taxi rises from 2,008s to 4,312s. The reason is
in the last column. `AOBT_3_flt` is the first ADS-B *movement*; the target is measured from
*off-block*. A flight that pushes back and then holds on the apron is airborne long after its
block time but only starts moving at the end of the hold, so the ADS-B proxy misses the entire
hold. `proxy_taxi_eobt1` tracks the target through the tail precisely because a filed estimate
cannot miss a hold it never observed.

Everything needed to detect this is already in the file, but **not in a form a tree can use**:
the hold shows up as a *difference* between two timestamps, and an axis-aligned split on each
proxy separately cannot express `IOBT - LOBT > 900`. `offblock_plan_spread` gives the range
and `offblock_delay_sec` one of the ten pairs; the other nine signed gaps are absent.

`LOBT - IOBT` is the one to watch: LOBT is revised by flow management, so that difference is
the closest thing in the file to an ATFM slot delay -- the usual reason an aircraft is held.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.features import model_features
from src.postprocess import (apply_extreme_proxy_override, apply_regime_model, fit_regime_model)
from src.validation import mirrored_split, ranking_composition, report

#: the five off-block estimates, in the order they are produced operationally
OFFBLOCK = {"sched": "SCHED_TIME_UTC_mvt", "iobt": "IOBT_flt", "lobt": "LOBT_flt",
            "eobt1": "EOBT_1_flt", "aobt3": "AOBT_3_flt"}


def add_gap_features(d: pd.DataFrame) -> list[str]:
    """Every signed pairwise gap among the off-block estimates, plus arrival-side analogues.

    Added in place; returns the new column names. All in seconds, all signed: the sign says
    which estimate is later, which is exactly the information a hold produces.
    """
    new: list[str] = []
    keys = list(OFFBLOCK)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            name = f"gap_{b}_{a}"
            d[name] = (d[OFFBLOCK[b]] - d[OFFBLOCK[a]]).dt.total_seconds()
            new.append(name)
    # the most and least pessimistic off-block estimate available for this flight
    stack = pd.concat([d[c] for c in OFFBLOCK.values()], axis=1)
    d["offblock_earliest"] = (d["MVT_TIME_UTC_mvt"] - stack.max(axis=1)).dt.total_seconds()
    d["offblock_latest"] = (d["MVT_TIME_UTC_mvt"] - stack.min(axis=1)).dt.total_seconds()
    d["offblock_n_present"] = stack.notna().sum(axis=1).astype("float32")
    new += ["offblock_earliest", "offblock_latest", "offblock_n_present"]
    # arrival-side: planned vs flown block-to-block, and how far the flown trajectory's two
    # endpoints disagree with the plan. A record whose ARVT_3 is off is one whose AOBT_3 may be.
    d["gap_arvt3_arvt1"] = (d["ARVT_3_flt"] - d["ARVT_1_flt"]).dt.total_seconds()
    d["blocktime_planned"] = (d["ARVT_1_flt"] - d["EOBT_1_flt"]).dt.total_seconds()
    d["blocktime_flown"] = (d["ARVT_3_flt"] - d["AOBT_3_flt"]).dt.total_seconds()
    d["blocktime_excess"] = d["blocktime_flown"] - d["blocktime_planned"]
    new += ["gap_arvt3_arvt1", "blocktime_planned", "blocktime_flown", "blocktime_excess"]
    return new


def run(feats, tr, val, d, is_train, y, yv, comp, vcell, trimmed, params, label):
    t0 = time.time()
    m = lgb.LGBMRegressor(**params).fit(tr[feats], y)
    pred = m.predict(val[feats])
    pred, _ = apply_regime_model(val, pred, fit_regime_model(d[is_train], feats))
    pred, _ = apply_extreme_proxy_override(val, pred)
    r = report(yv, pred, val, comp, vcell, trimmed_mask=trimmed)
    print(f"{label:<40} full {r['rmse_full']:7.2f}  trimmed {r['rmse_trimmed']:7.2f}  "
          f"({time.time()-t0:.0f}s, {len(feats)} feats)", flush=True)
    return r, m, pred


def main() -> None:
    raw = json.loads(Path("data/processed/lgbm_best_params_r3.json").read_text())
    INT_P = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}
    p = {k: (int(v) if k in INT_P else v) for k, v in raw.items() if not k.startswith("_")}
    n_est = p.pop("n_estimators")
    params = dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1,
                  subsample_freq=1, random_state=0, n_estimators=n_est, **p)

    comp = ranking_composition(pd.read_parquet(
        "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
    d = pd.read_parquet("data/processed/train_features.parquet")
    d = d.merge(pd.read_parquet("data/processed/train_surface.parquet"),
                on="MVT_ID_mvt", how="left", validate="one_to_one")
    gaps = add_gap_features(d)
    print(f"added {len(gaps)} gap features: {', '.join(gaps)}\n")

    nums, cats = model_features(d)
    for c in cats:
        d[c] = d[c].astype("string").fillna("__NA__").astype("category")
    # `model_features` selects by exclusion, so it has already absorbed the new columns --
    # the baseline has to remove them explicitly or the "with" arm duplicates every one.
    base_feats = [c for c in nums + cats if c not in set(gaps)]

    is_train, is_val, cell = mirrored_split(d)
    tr, val = d[is_train & trainable_mask(d)], d[is_val].copy()
    y, yv = tr["TAXITIME_SEC_mvt"], val["TAXITIME_SEC_mvt"].to_numpy()
    vcell, trimmed = cell[is_val], trainable_mask(val)
    print(f"fit {len(tr):,} | val {len(val):,} | {n_est} iterations\n")

    r0, _, pred0 = run(base_feats, tr, val, d, is_train, y, yv, comp, vcell, trimmed,
                       params, "baseline (127 features)")
    r1, m1, pred1 = run(base_feats + gaps, tr, val, d, is_train, y, yv, comp, vcell, trimmed,
                        params, f"+ {len(gaps)} off-block gaps")
    print(f"\ndelta: {r1['rmse_full'] - r0['rmse_full']:+.2f}s full   "
          f"{r1['rmse_trimmed'] - r0['rmse_trimmed']:+.2f}s trimmed")

    imp = pd.Series(m1.booster_.feature_importance("gain"), index=base_feats + gaps)
    imp = 100 * imp / imp.sum()
    print(f"\nnew features' share of total split gain: {imp[gaps].sum():.2f}%")
    print(imp[gaps].sort_values(ascending=False).round(3).to_string())
    print(f"\nrank of each new feature among all {len(imp)}:")
    order = imp.rank(ascending=False).astype(int)
    print(order[gaps].sort_values().to_string())

    # did it land where the theory said it would -- the long-taxi tail?
    normal = ~(val.dq_no_aircraft_type | val.dq_implausible_target).to_numpy()
    print("\nRMSE on normal rows by true-taxi band, before -> after:")
    for lo, hi in [(0, 1200), (1200, 1800), (1800, 2700), (2700, 3600), (3600, 1e9)]:
        m = normal & (yv >= lo) & (yv < hi)
        a = np.sqrt(((pred0[m] - yv[m]) ** 2).mean())
        b = np.sqrt(((pred1[m] - yv[m]) ** 2).mean())
        print(f"  [{lo:>5},{hi:>7.0f})  n={int(m.sum()):>7,}  {a:8.1f} -> {b:8.1f}  ({b-a:+7.1f})")
    np.save("data/processed/exp_cache/r4_pred_gaps.npy", pred1)


if __name__ == "__main__":
    main()
