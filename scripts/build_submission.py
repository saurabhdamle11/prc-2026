#!/usr/bin/env python
"""Fit on all 12 months and write a submission file in `submitting.parquet` format.

    python scripts/build_submission.py --team my-team-name --version 1

Differs from `scripts/train_baseline.py` in one deliberate way: that script fits on Jan-Jun
only, because its holdout has to stay forward in time. This one fits on **all 2,085,047 DEP
rows**, so the validated 320.7 s is a pessimistic estimate of what this model does.

Two post-processing rules are applied after the regressor, both documented because neither is
something the model can learn for itself:

1. **The missing-aircraft-type regime model.** Rows with a null `AIRCRAFT_TYPE_mvt` are a separate
   regime (training mean ~6,450 s against ~990 s for normal rows). They are excluded from the
   fit by `trainable_mask`, so a tree never sees a 7,000 s target and cannot extrapolate to
   one — left alone it predicts ~1,900 s for them. Overriding with the training conditional
   mean is worth **18.5 s** of weighted RMSE, measured in `SKILLS.md` §5.

   Note what this rule keys on: `AIRCRAFT_TYPE_mvt` being null, which is visible at
   prediction time. It deliberately does *not* use `trainable_mask`, which also catches
   implausible targets — that needs the label and is unavailable here. Measuring the branch
   with the mask instead overstates it by 5 s (23.7 s vs the honest 18.5 s).

2. **A floor on predictions.** RMSE is unbounded below but taxi time is not; a negative or
   near-zero prediction is always wrong. Clipped at the 0.1st percentile of observed
   trainable taxi times.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning import submitting_path, trainable_mask  # noqa: E402
from src.config_features import CONFIG_CATEGORICALS  # noqa: E402
from src.features import model_features  # noqa: E402
from src.offblock_gaps import add_offblock_gaps  # noqa: E402
from src.postprocess import (  # noqa: E402
    apply_extreme_proxy_override,
    apply_regime_model,
    fit_regime_model,
)

#: Fixed rather than early-stopped: the all-data fit has no held-out slice that is not
#: already part of the validation protocol. Defaults to the ``n_estimators`` recorded in the
#: parameter file, which the round-3 sweep chose by early stopping on an inner split; the
#: round-1 file has none, so 1400 stands in for it there.
N_ITERATIONS_FALLBACK = 1400

#: The round-3 sweep's result. It beats the round-1 file by only 1.7 s on the mirrored
#: protocol -- inside the ±6 s the protocol is good for (`SKILLS.md` §7) -- but it is the
#: better of the two on both the full and the trimmed number, and it was tuned against the
#: 127-feature model rather than the 93-feature one the round-1 file saw.
TUNED_PATH = Path("data/processed/lgbm_best_params_r3.json")

#: Prefixes of the round-4 configuration and rotation features (`src/config_features.py`).
#: Kept out of the regime branch: that model is fitted on ~1,500 rows, and the arm measured in
#: `SKILLS.md` §11 did not give it these columns.
CONFIG_PREFIXES = ("depcfg_", "arrcfg_", "stand_turnaround", "stand_prev_arr")

OUT_DIR = Path("submissions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="team name, e.g. 'youthful-giraffe'")
    ap.add_argument("--version", type=int, required=True, help="submission version number")
    ap.add_argument("--iterations", type=int, default=None,
                    help="boosting rounds; default is the parameter file's n_estimators")
    ap.add_argument("--params", type=Path, default=TUNED_PATH,
                    help="LightGBM parameter JSON to fit with")
    ap.add_argument("--seeds", type=int, default=3,
                    help="fits to average. Seed bagging cuts variance without touching bias; "
                         "3 seeds measured -0.84 s full / -1.00 s trimmed (SKILLS.md §11)")
    ap.add_argument("--cat-smooth", type=float, default=100.0,
                    help="LightGBM categorical smoothing. 100 measured best on a unimodal "
                         "curve (1->280.47, 10->279.82, 50->279.35, 100->279.15, 200->279.76, "
                         "400->279.96); the sweeps never touched this axis. SKILLS.md §11")
    ap.add_argument("--no-config", action="store_true",
                    help="skip the round-4 runway-configuration and stand-rotation features "
                         "(costs ~1.4 s full / ~1.9 s trimmed)")
    ap.add_argument("--no-extras", action="store_true",
                    help="skip the v5 additions: off-block gap features and OSM taxi geometry "
                         "(together worth ~1.4 s full / ~1.5 s trimmed; SKILLS.md §11)")
    ap.add_argument("--no-override", action="store_true",
                    help="skip the corroborated-extreme-proxy rule (the high-variance one). "
                         "Costs ~36 s of validated RMSE; use only to hedge that specific bet.")
    args = ap.parse_args()

    print("loading feature tables...")
    def _load(label: str) -> pd.DataFrame:
        """Feature table joined to its rolling surface-state companion.

        `validate="one_to_one"` is the guard that matters: a silent fan-out here would change
        the row count and the submission would be rejected for a mismatched ID set.
        """
        base = pd.read_parquet(f"data/processed/{label}_features.parquet")
        surf = pd.read_parquet(f"data/processed/{label}_surface.parquet")
        out = base.merge(surf, on="MVT_ID_mvt", how="left", validate="one_to_one")
        assert len(out) == len(base), f"{label}: surface join changed the row count"
        if not args.no_extras:
            osm_path = Path(f"data/processed/{label}_osm.parquet")
            if not osm_path.exists():
                raise SystemExit(f"{osm_path} not found — run "
                                 "scripts/build_osm_features.py, or pass --no-extras")
            out = out.merge(pd.read_parquet(osm_path), on="MVT_ID_mvt", how="left",
                            validate="one_to_one")
            assert len(out) == len(base), f"{label}: OSM join changed the row count"
        if not args.no_config:
            cfg_path = Path(f"data/processed/{label}_config.parquet")
            if not cfg_path.exists():
                raise SystemExit(f"{cfg_path} not found — run "
                                 "scripts/build_config_features.py, or pass --no-config")
            out = out.merge(pd.read_parquet(cfg_path), on="MVT_ID_mvt", how="left",
                            validate="one_to_one")
            assert len(out) == len(base), f"{label}: config join changed the row count"
        return out

    train, rank = _load("train"), _load("ranking")
    if not args.no_extras:
        # computed rather than cached: they are pure column arithmetic on timestamps already
        # present in both feature tables, so there is nothing to build or keep in sync
        train, gap_cols = add_offblock_gaps(train)
        rank, _ = add_offblock_gaps(rank)
        print(f"  + {len(gap_cols)} off-block gap features, + OSM taxi geometry")
    nums, cats = model_features(train)
    cats = list(cats) + [c for c in CONFIG_CATEGORICALS if c in train.columns]
    # LightGBM encodes pandas categoricals by their integer codes, so train and ranking MUST
    # share one category mapping. Built separately they would silently disagree — the same
    # airline could be code 3 in one frame and code 7 in the other, and the predictions would
    # be quietly wrong rather than erroring.
    for c in cats:
        train[c] = train[c].astype("string").fillna("__NA__")
        rank[c] = rank[c].astype("string").fillna("__NA__")
        levels = pd.Index(sorted(set(train[c].unique()) | set(rank[c].unique())))
        dtype = pd.CategoricalDtype(categories=levels)
        train[c] = train[c].astype(dtype)
        rank[c] = rank[c].astype(dtype)
    feats = nums + cats

    fit = train[trainable_mask(train)]
    print(f"  fitting on {len(fit):,} of {len(train):,} DEP rows (all 12 months)")

    int_params = {"num_leaves", "min_child_samples", "max_bin", "max_depth"}
    raw_params = json.loads(args.params.read_text())
    tuned = {k: (int(v) if k in int_params else v) for k, v in raw_params.items()
             if not k.startswith("_") and k != "n_estimators"}
    tuned["cat_smooth"] = args.cat_smooth
    iterations = args.iterations or int(raw_params.get("n_estimators", N_ITERATIONS_FALLBACK))
    print(f"  tuned params from {args.params}, {iterations} iterations, "
          f"{args.seeds} seed(s), cat_smooth={args.cat_smooth}, {len(feats)} features")

    # Seed bagging: the fits differ only in `random_state`, so averaging them reduces variance
    # without shifting the conditional mean the metric asks for.
    preds = []
    for seed in range(args.seeds):
        model = lgb.LGBMRegressor(objective="regression", verbose=-1, n_jobs=-1,
                                  subsample_freq=1, n_estimators=iterations,
                                  random_state=seed, **tuned)
        model.fit(fit[feats], fit["TAXITIME_SEC_mvt"])
        preds.append(model.predict(rank[feats]))
        print(f"    seed {seed}: mean {preds[-1].mean():,.0f}s")
    pred = np.mean(preds, axis=0)
    print(f"\nraw predictions: mean {pred.mean():,.0f}s  min {pred.min():,.0f}s  max {pred.max():,.0f}s")

    # --- rule 1: the missing-aircraft-type regime ------------------------------------
    branch = fit_regime_model(train, [c for c in feats if not c.startswith(CONFIG_PREFIXES)])
    is_regime = rank["dq_no_aircraft_type"].to_numpy()
    before = pred[is_regime].mean() if is_regime.any() else float("nan")
    pred, n_changed = apply_regime_model(rank, pred, branch)
    print(f"\nregime model: {n_changed} rows with a null AIRCRAFT_TYPE_mvt")
    print(f"  LightGBM on (proxy_taxi_sched - taxi), fitted on {branch['n_fitted']} rows"
          f" using {len(branch['features'])} populated features")
    print(f"  model predicted mean {before:,.0f}s -> branch mean {pred[is_regime].mean():,.0f}s")

    # --- rule 2: corroborated implausible proxy ---------------------------------------
    n_ov = 0
    if args.no_override:
        print("\nextreme-proxy override: SKIPPED (--no-override)")
    else:
        pred, n_ov = apply_extreme_proxy_override(rank, pred)
    if n_ov:
        print(f"\nextreme-proxy override: {n_ov} rows where AOBT_3 is implausible AND corroborated")
        m = ((rank["proxy_taxi_aobt3"] > 7200) & (rank["offblock_agree_n"] >= 2)).to_numpy()
        print(rank.loc[m, ["ADEP_mvt", "proxy_taxi_aobt3", "offblock_plan_median",
                           "offblock_agree_n"]].assign(pred=pred[m]).round(0).to_string(index=False))

    # --- rule 3: floor ----------------------------------------------------------------
    floor = float(np.percentile(fit["TAXITIME_SEC_mvt"], 0.1))
    n_below = int((pred < floor).sum())
    pred = np.maximum(pred, floor)
    print(f"floor at {floor:,.0f}s (0.1st pct of trainable targets): {n_below} predictions raised")

    # --- assemble in the exact submitting.parquet shape --------------------------------
    template = pd.read_parquet(submitting_path())
    out = (pd.DataFrame({"MVT_ID_mvt": rank["MVT_ID_mvt"].to_numpy(), "TAXITIME_SEC_mvt": pred})
             .set_index("MVT_ID_mvt")
             .reindex(template["MVT_ID_mvt"])          # template's exact order
             .reset_index())
    out["MVT_ID_mvt"] = out["MVT_ID_mvt"].astype(template["MVT_ID_mvt"].dtype)
    out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].astype("float64")

    # --- SKILLS.md section 7 sanity checks ---------------------------------------------
    print("\nsanity checks")
    checks = {
        "row count matches template": len(out) == len(template),
        "ID set matches template exactly": set(out["MVT_ID_mvt"]) == set(template["MVT_ID_mvt"]),
        "ID order matches template": out["MVT_ID_mvt"].equals(template["MVT_ID_mvt"]),
        "columns match template": list(out.columns) == list(template.columns),
        "no NaN predictions": out["TAXITIME_SEC_mvt"].notna().all(),
        "no negative predictions": (out["TAXITIME_SEC_mvt"] >= 0).all(),
        "dtypes match template": out.dtypes.equals(template.dtypes),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise SystemExit("submission failed its sanity checks — not written")

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{args.team}_v{args.version}.parquet"
    out.to_parquet(path, index=False)
    model.booster_.save_model("data/processed/lgbm_full.txt")
    print(f"\nwrote {path}  ({len(out):,} rows)")
    print(out["TAXITIME_SEC_mvt"].describe().to_string())


if __name__ == "__main__":
    main()
