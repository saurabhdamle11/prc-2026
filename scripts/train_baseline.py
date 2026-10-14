#!/usr/bin/env python
"""Train the model through the mirrored validation protocol and report honest numbers.

    python scripts/train_baseline.py

Unlike `scripts/smoke_test.py` (a plain time split), this uses `src/validation.py`: train on
Jan-Jun 2025, validate on Dec 2025 (all ten airports, standing in for January) plus Jul 2025
(restricted to EDDF/EGLL/EHAM), with each (slice, airport) cell re-weighted onto the share the
leaderboard actually gives it.

Two rules the protocol enforces and this script must not quietly break:
  * the model FITS on `trainable_mask` rows only, but is SCORED on every validation row —
    the ranking set contains 108 anomalous rows and the leaderboard scores all of them;
  * baselines are scored through the identical weighting, otherwise the comparison is
    between a model and a differently-measured baseline, which is how the old
    "519.5 s vs 385 s" confusion arose in the first place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning import MAX_PLAUSIBLE_TAXI_SEC, trainable_mask  # noqa: E402
from src.features import model_features  # noqa: E402
from src.validation import mirrored_split, naive_time_split, ranking_composition, report  # noqa: E402


def main() -> None:
    comp = ranking_composition(
        pd.read_parquet("Dataset/ranking.parquet",
                        columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"])
    )

    d = pd.read_parquet("data/processed/train_features.parquet")
    nums, cats = model_features(d)
    for c in cats:
        d[c] = d[c].astype("string").fillna("__NA__")
    feats = nums + cats

    is_train, is_val, cell = mirrored_split(d)
    fit_mask = is_train & trainable_mask(d)          # fit on trustworthy labels only
    val = d[is_val].copy()                            # score on everything
    val_cell = cell[is_val]

    tr = d[fit_mask]
    y, yv = tr["TAXITIME_SEC_mvt"].to_numpy(), val["TAXITIME_SEC_mvt"].to_numpy()
    print(f"fit rows   : {len(tr):,}  (Jan-Jun 2025, trainable only)")
    print(f"val rows   : {len(val):,}  (Dec 2025 all airports + Jul 2025 at EDDF/EGLL/EHAM)")
    print(f"val anomalous rows kept: {int((~trainable_mask(val)).sum()):,}\n")

    trimmed = trainable_mask(val)

    def show(name, pred):
        r = report(yv, np.asarray(pred), val, comp, val_cell, trimmed_mask=trimmed)
        print(f"{name:<34} weighted RMSE {r['rmse_full']:8.1f}s   "
              f"trimmed {r['rmse_trimmed']:7.1f}s   (unweighted {r['rmse_unweighted']:.1f}s)")
        return r

    med = tr.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].median()
    show("baseline: per-airport median", val["ADEP_mvt"].map(med).to_numpy())
    proxy = (val["MVT_TIME_UTC_mvt"] - val["AOBT_3_flt"]).dt.total_seconds()
    show("baseline: raw AOBT_3 proxy", proxy.fillna(val["ADEP_mvt"].map(med)).to_numpy())

    ci = [feats.index(c) for c in cats]
    model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=8, loss_function="RMSE",
                              random_seed=0, verbose=100, early_stopping_rounds=50, thread_count=-1)
    model.fit(Pool(tr[feats], y, cat_features=ci),
              eval_set=Pool(val[feats], yv, cat_features=ci), use_best_model=True)
    pred = model.predict(val[feats])
    res = show("\nCatBoost (88 features)", pred)

    print("\n=== per-airport breakdown (bias = mean signed error, + means over-predicting) ===")
    print(res["per_airport"].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # For contrast: the plain time split the smoke test used, same model settings.
    nt_tr, nt_va = naive_time_split(d)
    ntr, nva = d[nt_tr & trainable_mask(d)], d[nt_va]
    m2 = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=8, loss_function="RMSE",
                           random_seed=0, verbose=0, early_stopping_rounds=50, thread_count=-1)
    m2.fit(Pool(ntr[feats], ntr["TAXITIME_SEC_mvt"], cat_features=ci),
           eval_set=Pool(nva[feats], nva["TAXITIME_SEC_mvt"], cat_features=ci), use_best_model=True)
    p2 = m2.predict(nva[feats])
    flat = float(np.sqrt(np.mean((p2 - nva["TAXITIME_SEC_mvt"].to_numpy()) ** 2)))
    print(f"\nreference: naive Jan-Oct/Nov-Dec split, unweighted RMSE {flat:.1f}s "
          f"(what the smoke test measured)")

    model.save_model("data/processed/catboost_mirrored.cbm")
    print("\nmodel -> data/processed/catboost_mirrored.cbm")


if __name__ == "__main__":
    main()
