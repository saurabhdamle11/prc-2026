#!/usr/bin/env python
"""Optuna sweep over LightGBM, optimising the mirrored-protocol weighted RMSE.

    python scripts/tune_lgbm.py --trials 60

Optimises the **actual contest objective**: weighted RMSE over the scored (slice, airport)
cells, with the shipped post-processing applied, scoring every validation row including the
anomalous ones. Not a plain holdout RMSE, which would rank trials on a distribution the
leaderboard never looks at.

LightGBM was cross-checked against CatBoost on this identical split first (337.3 s vs 339.3 s
full, 219.9 s vs 221.5 s trimmed) — see `scripts/train_lgbm.py`. A sweep on a library that
disagreed with the reference would optimise the wrong thing.

Round-3 rewrite (2026-09-19) — the previous version had three defects, all of which made its
output the wrong answer to the right question:

1. **It never merged ``train_surface.parquet``**, so it swept a **93**-feature model while the
   shipped one uses **127**. Every rolling surface-state feature was invisible to the sweep,
   and the parameters it chose were tuned for a model that is not the one being submitted.
   This is why ``lgbm_best_params.json`` is described as an 88/93-feature result.
2. **It early-stopped on the slice it scored.** The iteration count was chosen using the same
   rows the trial was graded on, so every number it printed was optimistic. Stopping now
   happens on an **inner split** — the last training month — which the score never touches.
3. **It applied the superseded v2 regime branch** (a constant conditional mean) rather than
   the shipped delta regime model and extreme-proxy override, so its absolute numbers were not
   comparable to the protocol's. Both shipped rules are now applied.

Defect 3 does not change how trials *rank* — both post-processing rules overwrite their rows
with values that do not depend on the sweep's hyperparameters (``_REGIME_PARAMS`` is fixed and
the override is rule-based), so they contribute a constant to the weighted MSE. It is fixed so
the printed number can be compared against the protocol's 281.5 s directly.

**On overfitting the validation set.** With ~60 trials against a fixed holdout some selection
bias is unavoidable — confirm any gain survives on the trimmed number too, not just the full
one, and remember the protocol is only good to ±6 s (see ``SKILLS.md`` §7). The sweep is
seeded with the incumbent parameters as trial 0, so the study can always be compared against
what currently ships rather than against an arbitrary starting point.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning import trainable_mask  # noqa: E402
from src.features import model_features  # noqa: E402
from src.postprocess import (  # noqa: E402
    apply_extreme_proxy_override,
    apply_regime_model,
    fit_regime_model,
)
from src.validation import (  # noqa: E402
    TRAIN_MONTHS,
    mirrored_split,
    ranking_composition,
    report,
)

TUNED_PATH = Path("data/processed/lgbm_best_params.json")

#: The round-1 result, enqueued as trial 0 so the sweep starts from what currently ships.
#: Three of these sit on the old search-space boundary (``learning_rate`` at the 0.02 floor,
#: ``min_child_samples`` near the 10 floor, ``max_bin`` at the top of the grid), which is why
#: the space below is wider than round 1's on exactly those axes.
INCUMBENT = dict(
    learning_rate=0.0245929435, num_leaves=464, min_child_samples=14,
    colsample_bytree=0.4985357566, subsample=0.82801499,
    reg_lambda=19.3573292607, reg_alpha=5.6135101845, max_bin=511,
)


def load_frame() -> pd.DataFrame:
    """Train features joined to the rolling surface state — the full 127-feature table."""
    d = pd.read_parquet("data/processed/train_features.parquet")
    surf = Path("data/processed/train_surface.parquet")
    if not surf.exists():
        raise SystemExit(
            f"{surf} not found — run scripts/build_surface_features.py first.\n"
            "Sweeping without it tunes a 93-feature model and is the bug this rewrite fixed."
        )
    n0 = len(d)
    d = d.merge(pd.read_parquet(surf), on="MVT_ID_mvt", how="left", validate="one_to_one")
    assert len(d) == n0, "surface join changed the row count"
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--max-iter", type=int, default=4000,
                    help="cap on boosting rounds before early stopping intervenes")
    ap.add_argument("--patience", type=int, default=75)
    ap.add_argument("--no-refit", action="store_true",
                    help="score the early-stopped model directly instead of refitting on the "
                         "full training window. Halves the cost per trial, but the scored "
                         "model has then seen 5 of 6 training months, not 6.")
    ap.add_argument("--out", type=Path, default=Path("data/processed/lgbm_best_params_r3.json"),
                    help="written here, NOT to lgbm_best_params.json — promote explicitly "
                         "once the result is verified through scripts/train_lgbm.py")
    args = ap.parse_args()

    comp = ranking_composition(pd.read_parquet(
        "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
    d = load_frame()
    nums, cats = model_features(d)
    # one shared category mapping so train, inner-stop and val all encode identically
    for c in cats:
        d[c] = d[c].astype("string").fillna("__NA__").astype("category")
    feats = nums + cats

    is_train, is_val, cell = mirrored_split(d)
    tr, val = d[is_train & trainable_mask(d)], d[is_val].copy()
    y, yv = tr["TAXITIME_SEC_mvt"], val["TAXITIME_SEC_mvt"].to_numpy()
    val_cell, trimmed = cell[is_val], trainable_mask(val)

    # the stopping month comes out of the *training* window, so the scored slice is never
    # consulted while choosing an iteration count
    inner = (tr["MVT_TIME_UTC_mvt"].dt.month == max(TRAIN_MONTHS)).to_numpy()
    outer_X, outer_y = tr.loc[~inner, feats], y[~inner]
    inner_X, inner_y = tr.loc[inner, feats], y[inner]

    # both post-processing rules are independent of the sweep's hyperparameters, so the
    # regime model is fitted once rather than per trial
    regime = fit_regime_model(d[is_train], feats)

    print(f"fit {len(tr):,} ({int((~inner).sum()):,} outer + {int(inner.sum()):,} inner-stop) | "
          f"val {len(val):,} | {len(feats)} features ({len(nums)} num + {len(cats)} cat) | "
          f"{args.trials} trials | refit={'no' if args.no_refit else 'yes'}")

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = dict(
            objective="regression", metric="rmse", verbose=-1, n_jobs=-1,
            subsample_freq=1, random_state=0,
            learning_rate=trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 63, 1500, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 300, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            max_bin=trial.suggest_categorical("max_bin", [127, 255, 511, 1023]),
        )

        # stage 1 — choose the iteration count on the inner month
        stopper = lgb.LGBMRegressor(**{**params, "n_estimators": args.max_iter})
        stopper.fit(outer_X, outer_y, eval_set=[(inner_X, inner_y)], eval_metric="rmse",
                    callbacks=[lgb.early_stopping(args.patience, verbose=False)])
        best_iter = int(stopper.best_iteration_ or args.max_iter)
        trial.set_user_attr("best_iteration", best_iter)
        trial.set_user_attr("hit_cap", best_iter >= args.max_iter)

        # stage 2 — refit on the whole training window at that count, which is what ships
        if args.no_refit:
            model = stopper
        else:
            model = lgb.LGBMRegressor(**{**params, "n_estimators": best_iter})
            model.fit(tr[feats], y)

        pred = model.predict(val[feats])
        pred, _ = apply_regime_model(val, pred, regime)
        pred, _ = apply_extreme_proxy_override(val, pred)
        r = report(yv, pred, val, comp, val_cell, trimmed_mask=trimmed)
        trial.set_user_attr("rmse_trimmed", r["rmse_trimmed"])
        trial.set_user_attr("seconds", round(time.time() - t0, 1))
        return r["rmse_full"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial(INCUMBENT)   # trial 0 = what currently ships

    def log(study, trial):
        best = study.best_trial
        cap = " CAP" if trial.user_attrs.get("hit_cap") else ""
        print(f"trial {trial.number:>3}  full {trial.value:7.2f}s  "
              f"trimmed {trial.user_attrs.get('rmse_trimmed', float('nan')):7.2f}s  "
              f"iter {trial.user_attrs.get('best_iteration', 0):>4}{cap}  "
              f"{trial.user_attrs.get('seconds', 0):>5.0f}s  "
              f"| best {best.value:7.2f}s (trial {best.number})", flush=True)

    study.optimize(objective, n_trials=args.trials, callbacks=[log])

    inc = study.trials[0]
    print(f"\nincumbent (trial 0)         : {inc.value:.2f}s full / "
          f"{inc.user_attrs['rmse_trimmed']:.2f}s trimmed, {inc.user_attrs['best_iteration']} iters")
    print(f"best weighted RMSE (full)   : {study.best_value:.2f}s")
    print(f"     its trimmed RMSE       : {study.best_trial.user_attrs['rmse_trimmed']:.2f}s")
    print(f"     best_iteration         : {study.best_trial.user_attrs['best_iteration']}")
    print(f"     improvement on incumbent: {inc.value - study.best_value:+.2f}s full / "
          f"{inc.user_attrs['rmse_trimmed'] - study.best_trial.user_attrs['rmse_trimmed']:+.2f}s trimmed")
    print("\nbest params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    payload = {**study.best_params,
               "n_estimators": study.best_trial.user_attrs["best_iteration"],
               "_rmse_full": study.best_value,
               "_rmse_trimmed": study.best_trial.user_attrs["rmse_trimmed"],
               "_incumbent_rmse_full": inc.value,
               "_incumbent_rmse_trimmed": inc.user_attrs["rmse_trimmed"],
               "_n_features": len(feats),
               "_trials": args.trials,
               "_tuned_utc": pd.Timestamp.utcnow().isoformat()}
    args.out.write_text(json.dumps(payload, indent=2))
    study.trials_dataframe().to_csv("data/processed/lgbm_study.csv", index=False)
    print(f"\n-> {args.out} and data/processed/lgbm_study.csv")


if __name__ == "__main__":
    main()
