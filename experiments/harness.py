"""Shared experiment harness: one mirrored-protocol run, with switchable variants.

Everything is measured through `src.validation`'s composition weighting so numbers here are
directly comparable to the submission log in SKILLS.md.
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import trainable_mask
from src.features import model_features
from src.postprocess import fit_regime_branch, apply_regime_branch
from src.validation import mirrored_split, ranking_composition, weighted_rmse

#: Where experiments cache intermediate predictions. Lives under `data/processed/`, which is
#: gitignored and fully rebuildable, so nothing cached here is ever published. Override with
#: PRC_EXP_CACHE if you want it somewhere else.
CACHE = Path(os.environ.get("PRC_EXP_CACHE", "data/processed/exp_cache"))
CACHE.mkdir(parents=True, exist_ok=True)

TUNED = json.loads(Path("data/processed/lgbm_best_params.json").read_text())
INT_P = {"num_leaves", "min_child_samples", "max_bin", "max_depth", "n_estimators"}


def params(n_estimators: int = 1150, **over) -> dict:
    p = {k: (int(v) if k in INT_P else v) for k, v in TUNED.items()
         if not k.startswith("_") and k != "n_estimators"}
    return dict(objective="regression", metric="rmse", verbose=-1, n_jobs=-1,
                subsample_freq=1, n_estimators=n_estimators, **p, **over)


def load(surface: bool = True, extra_cats: bool = True) -> pd.DataFrame:
    d = pd.read_parquet("data/processed/train_features.parquet")
    if surface:
        s = pd.read_parquet("data/processed/train_surface.parquet")
        d = d.merge(s, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if extra_cats:
        d["stand_rwy"] = (d["stand_area"].astype("string") + ">"
                          + d["RUNWAY_mvt"].astype("string")).astype("string")
    return d


def split(d: pd.DataFrame):
    is_train, is_val, cell = mirrored_split(d)
    return is_train, is_val, cell


def featurelist(d: pd.DataFrame, drop: set[str] | None = None, extra_cats=("stand_rwy",)):
    nums, cats = model_features(d)
    cats = list(cats) + [c for c in extra_cats if c in d.columns and c not in cats]
    nums = [c for c in nums if c not in cats]
    if drop:
        nums = [c for c in nums if c not in drop]
        cats = [c for c in cats if c not in drop]
    return nums, cats


def encode(d: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    for c in cats:
        d[c] = d[c].astype("string").fillna("__NA__").astype("category")
    return d


class Scorer:
    """Weighted RMSE on the mirrored validation slice, in the three reported flavours."""

    def __init__(self, d, is_val, cell):
        self.comp = ranking_composition(pd.read_parquet(
            "Dataset/ranking.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]))
        self.val = d[is_val]
        self.y = self.val["TAXITIME_SEC_mvt"].to_numpy()
        self.cell = cell[is_val]
        self.apt = self.val["ADEP_mvt"]
        self.trim = trainable_mask(self.val).to_numpy()
        self.noact = self.val["dq_no_aircraft_type"].to_numpy()

    def _w(self, p, m=None):
        if m is None:
            return weighted_rmse(self.y, p, self.cell, self.apt, self.comp)
        return weighted_rmse(self.y[m], p[m], self.cell[m], self.apt[m], self.comp)

    def report(self, pred, branch=None, label=""):
        p = np.asarray(pred, dtype="float64")
        raw, trim = self._w(p), self._w(p, self.trim)
        line = f"{label:<34} raw {raw:7.1f}  trimmed {trim:7.1f}"
        if branch is not None:
            pb, _ = apply_regime_branch(self.val, p, branch)
            line += f"  +branch {self._w(pb):7.1f}"
            raw = self._w(pb)
        print(line, flush=True)
        return {"label": label, "full": raw, "trimmed": trim}


def fit_predict(tr, val, feats, y, p=None, init_tr=None, init_val=None, seed=0):
    m = lgb.LGBMRegressor(**(p or params()), random_state=seed)
    m.fit(tr[feats], y, init_score=init_tr)
    out = m.predict(val[feats], raw_score=init_val is not None)
    if init_val is not None:
        out = out + init_val
    return m, out
