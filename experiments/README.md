# Round-2 experiments (2026-09-13)

One script per hypothesis, each self-contained and re-runnable. Numbered in the order they were
run, which is also roughly the order the reasoning went. **The negative results are the point of
keeping these** — each one closes off a direction that looks attractive from the outside.

Run any of them from the repo root with the venv active:

    source .venv/bin/activate && python experiments/<script>.py

`harness.py` is shared scaffolding (load, mirrored split, weighted scoring, one LightGBM fit).

| script | question | answer |
|---|---|---|
| `budget.py` | where does the weighted MSE actually come from? | normal 53.5%, missing-aircraft-type 23.4%, invisible 23.1% |
| `anomaly_probe.py` | what *are* the broken-label rows? | two distinct populations; the invisible ones cluster at 2-3 h, not 24 h |
| `tail_probe.py` | is the 7,200 s training cut discarding signal? | the bands are a continuum; `proxy_taxi_sched` tracks the target in all of them |
| `identity_probe.py` | does `BLOCK_TIME` ever exactly equal a ranking-available field? | **no** — best is `AOBT_3_flt` at 0.65%, and unidentifiable in advance |
| `e1_surface.py` | do rolling surface-state features help? | **yes** — 242.8 → 238.3 trimmed |
| `e2_regime.py` | can the regime branch beat a 1-D linear fit? | **yes** — fit on `proxy_taxi_sched - y` instead of `y` |
| `e2b_regime_stability.py` | is that a fluke of one holdout? | wins 5 of 6 disjoint month splits |
| `e3_structure.py` | wider training mask / boosting from the proxy? | wider mask **no**; boost-from-proxy looked huge but see `e9` |
| `e4_clockgap.py` | can the arrival rows reveal the ADS-B/airport clock offset? | **no** — within-airport r = −0.04 |
| `e5_regime_structure.py` | is the regime bimodal, and is the mode predictable? | bimodal yes; `stand_area` separates it 5% → 64% |
| `e6_mixture.py` | so does a two-component mixture help? | **no** — worse on all 6 splits; even the mode-oracle is worse |
| `e7_keyed.py` | finer surface features (runway / apron / apron→runway)? | **no** — dead flat |
| `e8_duplicates.py` | do unmatched flights have a matched twin to join to? | **no** — 3.0%, and block times never agree |
| `e9_verify_boost.py` | where did `e3`'s 34.5 s actually come from? | **one row**; drop it and the variant is worse. The cautionary tale of this round |
| `e10_extreme_proxy.py` | when the proxy is implausible, is it right? | sometimes — and the scored set has only 5 such rows |
| `e11_shift_consistency.py` | can we tell *which* of those to trust? | **yes** — agreement among the other off-block estimates separates them 19/19 |
| `e12_corroboration.py` | does the same rule extend to the 1-2 h band (180x the exposure)? | **no** — the group mean beats the proxy there |
| `e13_final.py` | everything that survived, measured together | 331.1 → 281.5 |
| `e14_ensemble.py` | does blending CatBoost with LightGBM help? | see `SKILLS.md` §8 |
| `e15_override_check.py` | proxy, plan consensus, or midpoint for the override? | immaterial (1,761-2,153 vs 33,836 for abstaining) |
| `e16_rule_honesty.py` | was the override rule derivable without the validation months? | **yes** — 6/6 on Jan-Jun, then 3/3 out-of-sample |

## Round-4 experiments (2026-09-26)

Round 3 finished the hyperparameter sweep (verdict: 1.7 s, i.e. nothing — see `SKILLS.md` §10).
These re-measure where the error is now and test what is left. Findings in `SKILLS.md` §11.

| script | question | answer |
|---|---|---|
| `e19_budget_r4.py` | where is the weighted MSE under the *current* model? | normal 70.6%, regime 25.6%, invisible 3.8%; **`jul LIRF` alone is 35.6%** |
| `e21_lirf_july.py` | what is in that one dominant cell? | 337 regime rows carry 64% of it; Rome in July is genuinely seasonal, not a 2025 artifact |
| `e22_calibration.py` | is the long-taxi under-prediction recoverable? | **no** — isotonic fitted out of sample buys 0.01 s. But it found the mechanism: `proxy_taxi_aobt3` inverts in the tail |
| `e23_offblock_gaps.py` | do signed pairwise timestamp gaps help? | **−0.52 s** — the model already had it |
| `e24_regime_deep.py` | what *are* the regime rows? | ~all LIRF; `BLOCK − SCHED` is bimodal with a point mass at zero; flight-number donor matching fails |
| `e25_regime_mixture.py` | does a proper two-component mixture beat the delta model? | **no** — even the mode-oracle is worse (4,505 vs 4,240 s) |
| `e26_regime_gate.py` | does *gating* the delta model on a placeholder classifier help? | **no** — oracle gate is worth 9 s of total RMSE, realizable gates 0 s |
| `e20_config.py` | build runway-configuration and stand-rotation features | built; train/serve medians match within 1%, 0.06% unseen configs |
| `e27_config_model.py` | do those two families move the bulk model? and does seed bagging? | **yes** — config −1.36 s, bagging −0.84 s, both with trimmed moving more |
| `e28_linear_tree.py` | can piecewise-linear leaves follow a proxy past the training range? | **no** — catastrophic (506.8 vs 284.2), worst on the tail it targeted |
| `e29_categorical_params.py` | do LightGBM's categorical parameters help? | **`cat_smooth=100`**, −0.67 s, unimodal curve; `max_cat_threshold` is a red herring |
| `e30_final_r4.py` | the exact v4 configuration, measured end to end | **276.50 / 231.49**, filed and scored **282.23** |
| `e31_offblock_ceiling.py` | how accurate must a self-computed off-block be? | `AOBT_3_flt` is 422.9 s vs truth; need ~150-200 s to reach 224 |
| `e32_simulated_offblock.py` | what would such a detector actually be worth? | 150 s @ 90% cover -> **165 board**; 300 s or 70% cover -> ~201 |
| `e33_structured_error.py` | does that survive a detector that fails on holds like AOBT_3 does? | **yes** — worst case (2,000 s on holds, 70% cover) still 222.3 |
| `e34_detector_v1.py` | can an ADS-B pushback detector beat `AOBT_3_flt`? | **no** — 2,152.9 s vs 697.9 s. Aircraft are not transmitting at the stand; the off-block moment is unobservable for ~75-85% of departures |
| `e35_osm_match.py` | do OSM stand/runway refs join to the challenge data? | **yes** — 88.1% of departures, 100% at LIRF, 100% of runway thresholds |
| `e36_osm_model.py` | does OSM taxi distance help the model? | **−0.44 s** — the categoricals already encoded it |
| `e37_stand_bound.py` | is "can't push back before the stand is vacated" a usable cap? | **no** — circular: needs the previous arrival before *off-block*, the thing being predicted. Truth violates it on 9.6% of rows |
| `e39_embedding_mlp.py` | can an embedding MLP compete with LightGBM? | 284.20 alone (10.5 s worse) but residual corr only **0.9003** |
| `e40_blend.py` | does blending the MLP with LightGBM help? | **yes, −1.22 s** at w=0.75, and the optimum is flat (0.70-0.80 spans 0.06 s) |
| `e41_mlp_earlystop.py` | does a properly early-stopped MLP blend better? | **no** — 3.8 s better alone but corr rises to 0.9307 and the blend is 0.26 s *worse* |
| `e42_three_way.py` | does blending both MLPs beat either? | **no** — 0.03 s over the best two-way |
| `e43_residuals_loss_te.py` | where is v5 wrong, and do loss/target transforms or OOF target encoding help? (run on the Oracle VM, 4 threads, 1 seed; ref 274.40 / 230.92) | rows > 30 min are 4.4% of rows but **45% of MSE**; LIRF 42%, July 72%. OOF target encoding **−0.67 s** (sub-noise); Huber +1.4 / −0.1 s; winsorising +10.9 / +2.9 s; log1p + Duan smearing **+4.6 s**. All closed |
| `e44_lirf_per_airport.py` | is it LIRF-in-July specifically, is LIRF's label noisier, and do per-airport models or a LIRF-only cap help? (Oracle VM, 1 seed, vs e43 ref 274.40 / 230.92) | **yes**, jul LIRF is 35.5% of MSE (22.6 pts from 337 missing-type rows; 466 s RMSE even on normal rows). No regime change: smooth seasonal curve, runway mix stable. LIRF and LTFM `AOBT_3 − BLOCK` disagree >10 min on 12.6% / 17.7% of rows vs 1.6–6.7% elsewhere. Per-airport models **+1.6 s** (LIRF alone +6.2 s on its rows); LIRF cap p99.5 **+1.7 s**, p99.9 **+0.4 s**. All closed |
| `e45_metar_convective.py` | do METAR thunderstorm/convective features (IEM, public domain) help? (Oracle VM, 1 seed, vs ref 274.40 / 230.92) | convective-in-2h rows are 3.5% of rows / 8.7% of MSE, ceiling 7.8 s. Measured **−0.51 s full / −0.57 s trimmed**: convective-row RMSE 432 → 425, bias −40 → −17 s, rest unchanged. Real but sub-noise |
| `e46_noact_autopsy_weights.py` | what are the missing-type rows, do dumb baselines beat the regime branch, and do ADS-B-disagreement sample weights help? | failed NM joins (100% no flight record / no AOBT_3), labels to p99 81,032 s; branch 3,885 s RMSE vs 13,017 s for airport-hour median (**+145 s** total if substituted). Weights: \|gap\|>600 at LIRF/LTFM **+4.15 s**, negative-gap only +0.28 s. Closed |
| `e47_regime_branch_variants.py` | can the missing-type branch be improved — no placeholders, Huber/log inside it, peer-delay features from `_mvt`? | **no, all lose** on 12-month LOMO (base 4,230 s) and on the mirrored total: no-placeholder +3.5 / +24.1 s, Huber +0.3 / +6.4 s, log +121.6 s, peer delay +4.9 s. Also corrects the 47.9% placeholder figure (exact match is 2–12% in every month) |
