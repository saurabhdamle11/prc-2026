#!/usr/bin/env python
"""LightGBM through the same mirrored validation protocol, for cross-checking and tuning.

    python scripts/train_lgbm.py

Exists because CatBoost costs ~9 min per fit, which makes an optuna sweep impractical.
Before trusting a sweep built on LightGBM, this must reproduce CatBoost's number on the
identical split — a sweep on a library that disagrees would optimise the wrong thing.

Same rules as `train_baseline.py`: fit on `trainable_mask` rows only, score every validation
row, and apply the post-processing rules afterwards.

.. warning::
   This script used to early-stop on the *scored* holdout, which makes every number it prints
   optimistic — the iteration count is then chosen using the same rows the score is read from.
   It now fits a fixed number of iterations by default. ``--early-stop`` is still available and
   holds out the last training month for the purpose, never the validation slice.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning import trainable_mask  # noqa: E402
from src.features import model_features  # noqa: E402
from src.postprocess import (  # noqa: E402
    apply_extreme_proxy_override,
    apply_regime_branch,
    apply_regime_model,
    fit_regime_branch,
    fit_regime_model,
)
from src.validation import (  # noqa: E402
    TRAIN_MONTHS,
    mirrored_split,
    ranking_composition,
    report,
)

import argparse, json  # noqa: E402

PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.06, num_leaves=255,
              min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              n_estimators=2000, verbose=-1, n_jobs=-1)

TUNED_PATH = Path("data/processed/lgbm_best_params.json")


def load_params(use_tuned: bool) -> dict:
    """Hand-picked defaults, or the optuna result when asked for and present."""
    if not use_tuned:
        return dict(PARAMS)
    if not TUNED_PATH.exists():
        raise SystemExit(f"{TUNED_PATH} not found — run scripts/tune_lgbm.py first")
    raw = json.loads(TUNED_PATH.read_text())
    # JSON has no int type distinction — LightGBM rejects num_leaves="464.0"
    int_params = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}
    tuned = {k: (int(v) if k in int_params else v) for k, v in raw.items()
             if not k.startswith("_") and k != "n_estimators"}
    return dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1,
                n_estimators=3000, subsample_freq=1, **tuned)


def prepare(df, cats):
    for c in cats:
        df[c] = df[c].astype("category")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", action="store_true", help="use data/processed/lgbm_best_params.json")
    ap.add_argument("--iterations", type=int, default=1150,
                    help="fixed boosting rounds (the protocol default; see the module warning)")
    ap.add_argument("--early-stop", action="store_true",
                    help="early-stop on the last training month instead of a fixed count")
    args = ap.parse_args()
    params = load_params(args.tuned)
    print(f"params: {'tuned (optuna)' if args.tuned else 'hand-picked defaults'}")

    comp = ranking_composition(pd.read_parquet(
        "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
    d = pd.read_parquet("data/processed/train_features.parquet")
    surf = Path("data/processed/train_surface.parquet")
    if surf.exists():
        n0 = len(d)
        d = d.merge(pd.read_parquet(surf), on="MVT_ID_mvt", how="left", validate="one_to_one")
        assert len(d) == n0, "surface join changed the row count"
    else:
        print("WARNING: no train_surface.parquet — run scripts/build_surface_features.py")
    nums, cats = model_features(d)
    # one shared category mapping so train and val encode identically
    for c in cats:
        d[c] = d[c].astype("string").fillna("__NA__").astype("category")
    feats = nums + cats

    is_train, is_val, cell = mirrored_split(d)
    tr, val = d[is_train & trainable_mask(d)], d[is_val].copy()
    y, yv = tr["TAXITIME_SEC_mvt"], val["TAXITIME_SEC_mvt"]
    print(f"fit {len(tr):,} | val {len(val):,} | features {len(feats)}")

    t0 = time.time()
    if args.early_stop:
        # the stopping month comes out of the *training* window, so the validation slice is
        # never consulted while choosing the iteration count
        inner = tr["MVT_TIME_UTC_mvt"].dt.month == max(TRAIN_MONTHS)
        model = lgb.LGBMRegressor(**{**params, "n_estimators": 3000})
        model.fit(tr[~inner][feats], y[~inner],
                  eval_set=[(tr[inner][feats], y[inner])], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(200)])
        print(f"trained in {time.time() - t0:.0f}s, best iteration {model.best_iteration_} "
              f"(stopped on month {max(TRAIN_MONTHS)}, {int(inner.sum()):,} rows held out)")
    else:
        model = lgb.LGBMRegressor(**{**params, "n_estimators": args.iterations})
        model.fit(tr[feats], y)
        print(f"trained in {time.time() - t0:.0f}s, {args.iterations} fixed iterations")

    pred = model.predict(val[feats])
    trimmed = trainable_mask(val)
    cellv, y = cell[is_val], yv.to_numpy()
    rmse = lambda p: report(y, p, val, comp, cellv)["rmse_full"]
    print(f"\nLightGBM  weighted RMSE {rmse(pred):.1f}s   "
          f"trimmed {report(y, pred, val, comp, cellv, trimmed_mask=trimmed)['rmse_trimmed']:.1f}s")

    # the two post-processing rules, applied in the order they ship, each reported separately
    # so a regression in either is visible rather than absorbed into one total
    lin, _ = apply_regime_branch(val, pred, fit_regime_branch(d[is_train]))
    print(f"  + linear regime branch (v2)      {rmse(lin):.1f}s")
    dlt, n1 = apply_regime_model(val, pred, fit_regime_model(d[is_train], feats))
    print(f"  + delta regime model ({n1} rows)  {rmse(dlt):.1f}s")
    fin, n2 = apply_extreme_proxy_override(val, dlt)
    print(f"  + extreme-proxy override ({n2} rows) {rmse(fin):.1f}s")
    print("\n(reference, 2026-09-13: 326.2 / 317.6 / 281.5 with 127 features; "
          "trimmed 237.3. v2's shipped configuration scored 331.1 here and 329.76 on the board.)")

    model.booster_.save_model("data/processed/lgbm_mirrored.txt")


if __name__ == "__main__":
    main()
