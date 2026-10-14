#!/usr/bin/env python
"""Feature-set smoke test: does the built feature table beat the baselines at all?

    python scripts/smoke_test.py

**This is not the stage-7 validation protocol.** It uses a plain Jan-Oct / Nov-Dec 2025 time
split, whereas the real evaluation is Jan+Jul 2026 with a lopsided composition (70% January,
July restricted to EDDF/EGLL/EHAM) that a scoring split has to mirror. Treat the number here
as "the feature set is wired up and carries signal", not as an expected leaderboard score.

Uses CatBoost because `lightgbm` and `xgboost` do not currently load in this environment
(missing the OpenMP runtime — `brew install libomp`). Writes the feature importances to
`data/processed/smoke_importance.csv`, which `analysis.ipynb` §13.6 reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning import trainable_mask  # noqa: E402
from src.features import model_features  # noqa: E402


def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def main() -> None:
    d = pd.read_parquet("data/processed/train_features.parquet")
    d = d[trainable_mask(d)].copy()
    nums, cats = model_features(d)
    for c in cats:
        d[c] = d[c].astype("string").fillna("__NA__")

    month = d["MVT_TIME_UTC_mvt"].dt.month
    tr, va = d[month <= 10], d[month >= 11]
    feats = nums + cats
    y, yv = tr["TAXITIME_SEC_mvt"].to_numpy(), va["TAXITIME_SEC_mvt"].to_numpy()
    print(f"train {len(tr):,} | val {len(va):,} | features {len(feats)}")

    med = tr.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].median()
    print(f"baseline per-airport median : {rmse(va['ADEP_mvt'].map(med), yv):7.1f}s")
    proxy = (va["MVT_TIME_UTC_mvt"] - va["AOBT_3_flt"]).dt.total_seconds().fillna(va["ADEP_mvt"].map(med))
    print(f"baseline raw AOBT_3 proxy   : {rmse(proxy, yv):7.1f}s")

    ci = [feats.index(c) for c in cats]
    model = CatBoostRegressor(
        iterations=500, learning_rate=0.1, depth=8, loss_function="RMSE",
        random_seed=0, verbose=100, early_stopping_rounds=50, thread_count=-1,
    )
    model.fit(Pool(tr[feats], y, cat_features=ci),
              eval_set=Pool(va[feats], yv, cat_features=ci), use_best_model=True)
    print(f"\nCatBoost, {len(feats)} features : {rmse(model.predict(va[feats]), yv):7.1f}s")

    imp = pd.Series(model.get_feature_importance(), index=feats).sort_values(ascending=False)
    imp.index.name = "feature"
    imp.rename("importance").to_csv("data/processed/smoke_importance.csv")
    print("\ntop 15:\n" + imp.head(15).to_string())


if __name__ == "__main__":
    main()
