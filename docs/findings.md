# Critical findings

Measured, not assumed. These came out of a full pass over all twelve training files and the scoring
file, and they drive the whole design. Code comments and `SKILLS.md` cite them as "finding N".

Verified in
`analysis.ipynb` §12; the rules derived from them live in `src/cleaning.py`.

**1. The task is "infer the off-block time, given the takeoff time."**
`TAXITIME_SEC_mvt == MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt` exactly (100.000% match). `BLOCK_TIME_UTC_mvt` is
**100% blank in `ranking.parquet`**; `MVT_TIME_UTC_mvt` (actual takeoff) is **100% present**. Consequences:
- The takeoff time is a **legitimate, fully available feature**. Congestion/queue features and weather joins
  should be anchored on it — including "how many aircraft actually took off from this runway in the ±N minutes
  around this one", which is computable identically in train and ranking.
- Anything derived from `BLOCK_TIME_UTC_mvt` is **unusable**. This invalidates the `pushback_delay_min` feature
  in the notebook's original §8 (flagged inline there).

**2. `AOBT_3_flt` is the most informative single field — but it does not beat a median baseline on its own.**
ADS-B-derived actual off-block time, present in 98.52% of ranking DEP rows and 98.92% of train DEP rows. Its
bias is strongly airport-specific (LTFM +296 s, LIRF −118 s), which a model can correct.

Earlier versions of these notes quoted "RMSE ≈ 385 s for the raw proxy vs 519.5 s for the per-airport median" and concluded
the proxy was the bar to beat. **Those two numbers were computed on different row sets** — 385 s excluded the
pathological tail, 519.5 s included it. On identical rows the ordering reverses:

| rows | per-airport median | raw `MVT_TIME − AOBT_3` |
|---|---|---|
| all DEP (2,085,047) | **519.5 s** | 521.4 s |
| trainable & `AOBT_3` present (2,062,474) | **373.0 s** | 380.9 s |

Adding a per-airport bias correction to the proxy reaches 363.7 s — a ~9 s gain over the median baseline, not
the dominant effect the old framing implied. `AOBT_3_flt` still belongs in the model, but **the real bar is
373 s on trainable rows**, and most of the apparent advantage of the proxy was tail handling, not better
prediction of a normal taxi. `EOBT_1_flt` gives a weaker proxy (RMSE ≈ 677 s).

**3. A ~0.03% tail dominates the RMSE metric.** Against a per-airport-median baseline, the 584 DEP rows over 2 h
carry **46% of all squared error**; the 4,126 rows over 1 h carry 55%. These are date/recording errors (the
largest cluster at ~24 h), not long taxis.

**4. That tail has a crisp signature: missing `AIRCRAFT_TYPE_mvt`.** 75% of DEP rows over 2 h lack an aircraft
type vs 0.07% of rows overall — ~1000× enrichment, almost all at LIRF. It is a genuinely separate regime
(n = 1,507, mean 6,451 s / σ 11,170 s, vs 987 s / 432 s for normal rows). The current `ranking.parquet`
contains **388 of them** (383 LIRF, 4 LTFM, 1 LSZH) — the pre-2026-09-04 file had only 108 — so they must be
predicted, not ignored. Overriding them with the training conditional mean was measured at **18.5 s** of
weighted RMSE on the old composition, and matters more now that there are 3.6× as many.

**5. Scoring composition — RE-MEASURED 2026-09-07 after the organizers replaced the file.** (This is the
Jan/Jul *leaderboard* phase. The final phase scores `final_ranking.parquet`: 670,790 DEP rows — Jan 152,719 /
Feb 144,158 / Jun 181,791 / Jul 192,122 — of which these 344,841 are a strict subset.)
The scored set is **344,841 DEP rows**: January 152,719 (**44.3%**) and July 192,122 (**55.7%**), with **all ten
departure airports present in both months**.

> ⚠️ **`ranking.parquet` and `submitting.parquet` were replaced on 2026-09-04.** The earlier revision had only
> 215,876 DEP rows, was 70.7% January, and had *only* EDDF/EGLL/EHAM in July. A submission built from it is
> **rejected**: `"Submission is missing 128965 required rows present in truth. (No zero-fill.)"` — which is
> also the explanation for the leaderboard's `usedPairs: 344841`. All 215,876 old IDs survive into the new file
> as a subset, so no prediction work was wasted, but every composition-derived number had to be redone.
> The training files were *not* changed. Superseded copies are kept in
> `Dataset/_superseded_2026-09-07/` for diffing.
>
> **Check the bucket against local before trusting any composition claim**: `mc ls prc/prc-2026-datasets/`.

**6. `ranking.parquet` is a complete movement record, not a sample.** This is what makes the congestion
features legitimate — they count surrounding traffic, so they only transfer if traffic is observed at the
same rate on both sides. Measured: daily movements per airport in Jan-2026 match Jan-2025 within ±5%
(EHAM is the one outlier at 0.88, consistent with Schiphol's real capacity cap rather than sampling), the
DEP/ARR balance is exactly 50/50 in both files, and all 31 days of each month are present. Congestion
counts are therefore on the same scale in training and at prediction time.

Minor: 369 DEP rows have negative taxi times, all between −12 s and −1 s (clock artefacts, clipped to 0);
`RUNWAY_mvt` uses a literal `"NA"` string for unknown.
