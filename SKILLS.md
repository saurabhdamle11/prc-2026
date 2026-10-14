# Working playbook — PRC Data Challenge 2026

Concrete checklists for each stage of the pipeline described in `README.md`. Follow these when doing the actual
work (in `analysis.ipynb` or new notebooks/scripts), so results stay reproducible and eligible per the challenge
rules.

## 1. External data sourcing — vetting checklist

Before adding *any* external dataset, confirm and record all of these (e.g. as a markdown table or notebook cell):

- [ ] **License is open-source/open-data** (e.g. CC-BY, CC0, ODbL, public-domain government data). Reject anything
      requiring a paid key with restrictive redistribution terms, or with unclear/"research only" terms.
- [ ] **Source is documented** — official provider, stable URL, version/date of the snapshot used.
- [ ] **Join key** — how it maps onto our data (by ICAO airport + date/hour, typically).
- [ ] **Coverage** — does it actually cover our 11 airports and the Jan–Dec 2025 + Jan/Jul 2026 date range? Data
      that only covers training months and not the Jan/Jul 2026 prediction window is useless for real inference
      (we'd have no way to compute the same feature at prediction time) unless it's a static/reference dataset
      (e.g. airport layout) rather than a time-varying one.
- [ ] **Latency-honest** — could this feature realistically be known at scheduled-taxi time in production, or does
      it leak future information (e.g. actual weather *after* the flight taxied)?

Plausible candidate categories (verify license/coverage before use, don't assume): airport surface/runway-config
reference data, historical weather/METAR for the 11 ICAO codes, public holiday calendars, airport
distance-to-runway reference layouts. OpenSky Network's own open datasets are a natural first place to check given
they co-run this challenge. Log rejected sources too (with the reason) so we don't re-litigate them later.

### Survey done (2026-09-02)

Checked against the eligibility.html requirement that additional datasets be "openly available under an open
source license" — not just free-to-view.

| Source | Covers | License | Verdict |
|---|---|---|---|
| **Open-Meteo Historical Weather API** | Hourly weather (wind, precip, temp, snowfall, cloud cover) at any lat/lon, full 2025–2026 range | CC BY 4.0 (data), AGPLv3 (code) | **Use** — clean open license, well documented, covers our full date range including Jan/Jul 2026 (historical by the time of submission, not a forecast) |
| **python-holidays** (PyPI) | Public holidays by country — DE/GB/NL/ES/FR/IT/TR/CH cover our 11 airports | MIT | **Use** — trivial to add, safe license |
| **OurAirports** (`airports.csv`/`runways.csv`) | Static reference: elevation, runway count/length/heading for the 11 airports | Public Domain / Unlicense | Optional — safe but likely low marginal value since we already have `RUNWAY_mvt`/`STAND_mvt` |
| EUROCONTROL PRU data hub (`ansperformance.eu/data/`) — Airport Traffic, Taxi-out Additional Time, ATFM delays | Exactly on-topic (same taxi-out metric!), same host as the contest | Attribution + **non-commercial** + **no-modification** (per their disclaimer page) | **Flagged, not used** — fails the open-source-license bar despite being topically perfect. Don't build the model to depend on this without an explicit organizer ruling (ask on Discord) first |
| OPDI (`euctrl-pru/OPDI`) | Processed OpenSky ADS-B flight tracks/events | Code MIT; **data-product license unstated** | **Flagged, not used** — ambiguous, and redundant with movement data we already have |
| **IEM (Iowa Environmental Mesonet) global ASOS/METAR archive** (vetted 2026-09-27) | Raw METAR + SPECI text, ~half-hourly, all ten departure airports, 2024-12-28 → 2026-08-03 incl. both ranking months | **Public domain** — "The materials found on this website are in the public domain and may be used freely by anyone for any lawful purpose" ([disclaimer](https://mesonet.agron.iastate.edu/disclaimer.php)); attribution appreciated, not required | **Vetted, experiment only** — fetched by `scripts/fetch_metar.py`, features in `src/metar_features.py`, measured in `e45` (sub-noise, not shipped). If it ever ships, add it to the README licence table with the IEM attribution |
| NOAA NCEI Integrated Surface Database | Decoded hourly surface obs incl. present weather | US-gov, but the product page does not state terms for non-US stations (WMO Res. 40 caveats commonly attach) | **Not used** — IEM covers the same METARs with an explicit public-domain statement |
| Meteostat | Historical weather | CC BY-NC 4.0 | **Rejected** — NonCommercial clause isn't an open-source license |
| Raw OpenSky Network bulk/API data | Same underlying data our own training set is presumably derived from | Free for non-commercial research; restricted for "operational" use, license required for commercial use | **Rejected** as an external addition — ambiguous license fit, and redundant with what we already have |

**Note**: our own training data already lets us compute strong airport-load/congestion features (rolling counts of
scheduled/actual movements per airport, runway, time window) without any external dataset or licensing question at
all — prioritize those before leaning on external sources for the "airport procedures and load" factor from the
rationale.

## 2. Data cleaning — DONE (`src/cleaning.py`)

Implemented and demonstrated in `analysis.ipynb` §12.1. Rules applied, each grounded in a measurement:

- [x] **Negative targets** — 369 DEP rows, all between −12 s and −1 s (LSZH 298, LFPG 61). The magnitudes make
      these clock artefacts, so they are clipped to 0 and flagged `dq_was_negative`.
- [x] **Implausible targets** — flagged above 7,200 s (`MAX_PLAUSIBLE_TAXI_SEC`, the 99.997th percentile). Not
      real long taxis: the largest cluster sits at ~24 h, and where the independent ADS-B off-block time exists
      it disagrees by >1 h for 96 of 103 such rows.
- [x] **The `"NA"` runway token** — `RUNWAY_mvt` uses a literal `"NA"` string for unknown; converted to a real
      null so it does not silently become its own category.
- [x] **`_mvt` vs `_flt` mismatches preserved as features** (`dq_ades_mismatch`, `dq_actype_mismatch`,
      `dq_rule_mismatch`, `dq_diverted`) rather than reconciled — the organizers state the inconsistency is
      intentional. Agreement rates on train DEP: ADEP 100%, ADES 98.59%, aircraft type 99.66%, rule 99.93%.
- [x] **Unmatched flight-plan records collapse to one flag** — all 19 `_flt` columns share a single missingness
      pattern (1.08% of train DEP, 1.48% of ranking DEP); it is one event, not 19 independent gaps.
- [x] **Same function runs over `ranking.parquet`** — cleaning repairs and flags but never drops; row exclusion
      is a separate decision in `trainable_mask`, which must never be applied to the ranking set. There is an
      assertion in §12.1 that the ranking row count survives cleaning unchanged.

Excluding untrustworthy labels drops the training target's σ from **546 s to 413 s** while removing only 1,652
rows (0.079%) — the single highest-leverage cleaning step for an RMSE metric.

## 3. EDA checklist

(`analysis.ipynb` sections 1–11 already cover most of this — extend rather than duplicate.)

> **Method note — always control before judging a feature.** Raw marginals in this dataset are badly confounded
> by airport and time of day. The clearest example: taxi time appears *shorter* at visibility < 200 m (median
> 851 s) than in clear conditions (903 s), which would suggest fog speeds up taxiing. It does not — low
> visibility coincides with calm winter nights when traffic is light. Comparing each flight to the median of its
> own (airport × hour-of-day × month) cell reverses the sign and gives a clean monotonic 1.07× → 1.00× gradient.
> Use that controlled ratio to decide whether a feature carries signal; a raw correlation will mislead. Worked
> example in `analysis.ipynb` §12.2.

- [ ] Target distribution by phase (DEP vs ARR) and airport.
- [ ] Missingness pattern, esp. whether missingness correlates with the target (informative missingness) — it
      strongly does here: a missing `AIRCRAFT_TYPE_mvt` marks a ~6.5× longer-taxi regime.
- [ ] Temporal patterns: hour-of-day, day-of-week, month, and whether these interact with airport (note LTAI has
      no departures, so "Antalya seasonality" is not available — LTFM/LEBL are the seasonal-leisure cases).
- [ ] Categorical drivers: `ADEP_mvt`, `RUNWAY_mvt`, `STAND_mvt`, `AIRCRAFT_TYPE_mvt`, `MARKET_SEGMENT_flt`,
      `AIRCRAFT_OPERATOR_flt`.
- [ ] Delay/schedule-adherence features vs. taxi time (pushback delay, EOBT delay) — already scaffolded.
- [ ] **Train-vs-ranking covariate shift**, per airport and per time-of-day, not just in aggregate — since the
      ranking set is Jan+Jul 2026 only (winter + summer), check both seasons separately against the matching
      training months (Jan 2025, Jul 2025) as well as against the full-year training distribution.
- [ ] Fill in the "Summary of findings" section (already stubbed at the end of the notebook) once each point above
      has an answer — don't leave it as a template.

## 4. Feature engineering — DONE (`src/features.py`)

Built by `scripts/build_features.py` into `data/processed/{train,ranking}_features.parquet` (115 columns
each, ~22 s for both sides). `model_features()` returns the 88 model inputs as (numeric, categorical).

**The availability rule that governs everything here** (measured, see `docs/findings.md` finding 1): a feature is
usable only if it is populated in `ranking.parquet`. The single disqualifying column is **`BLOCK_TIME_UTC_mvt`,
which is 100% blank there** — the target is exactly `MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt`, so the off-block
time *is* the thing being predicted. `MVT_TIME_UTC_mvt` (actual takeoff) is 100% populated in the ranking set
and is therefore a legitimate feature, which inverts the usual intuition about what counts as leakage here.

### Measured signal

Every candidate was judged on the **controlled** comparison from §3 — median taxi relative to the flight's own
(airport × hour × month) cell, so 1.00 means "adds nothing beyond airport, time-of-day and season". Raw
marginals were not trusted.

| feature | controlled ratio, low → high | verdict |
|---|---|---|
| `planned_enroute_sec` | 0.874 → 1.131 | **strongest**, monotonic over 6 bins |
| `queue_at_pushback` | 0.963 → 1.095 | **strong**, monotonic over 7 bins |
| `wind_crosswind_ms` | 0.999 → 1.062 | modest, monotonic |
| `sec_since_prev_dep_rwy` | 0.946 → 1.021 | modest, **non-monotonic** (inverted U) |
| `cong_rwy_30` | 0.991 → 1.016 | weak — only the top bin moves |
| `cong_dep_60` | 0.994 → 1.008 | **essentially flat** |

**The window-count congestion family underdelivered and the queue formulation carried it.** This section
previously called takeoff-anchored congestion "the highest-value feature family still to build". Half right:
counting movements in a ±N-minute window is almost entirely absorbed by the airport × hour control, because a
count of nearby traffic is mostly a restatement of time-of-day. What works is measuring the **surface state**
instead — `queue_at_pushback` counts flights that have pushed back (`AOBT_3_flt`) but not yet taken off
(`MVT_TIME_UTC_mvt`) at the moment this aircraft pushes back, which is the queue it is physically joining.
Both timestamps survive into the ranking set, so it is computable at prediction time. Keep the window counts
(they are cheap and the tree may use them in interactions) but do not expect them to carry the model.

Sanity check on the queue feature, honestly reported: it gets the **extremes** right (EGLL highest on both
queue and taxi, LSZH lowest on both) but the rank correlation across all ten airports is only **0.396**. The
exceptions carry the lesson — LIRF has a *short* queue (6) with the second-longest taxi (1,025 s), EHAM a
middling queue (11) with the second-shortest (742 s). Taxi time is roughly *distance ÷ speed + queueing*, and
airports differ in **geometry** as much as in congestion: Fiumicino's long taxiways produce long taxis at low
traffic, while Schiphol's stands sit close to its runways. No congestion feature can separate those regimes,
which is why the stand/runway family ends up carrying ~25% of model importance.

### The families built

- **Off-block proxies** (`src/cleaning.add_offblock_proxies`): `MVT_TIME − AOBT_3_flt` and the weaker
  `EOBT_1_flt` / `IOBT_flt` / `LOBT_flt` variants, plus `offblock_delay_sec` and `takeoff_vs_sched_sec`.
- **Congestion / queue / sequence** (`add_congestion_features`, `add_queue_features`,
  `add_sequence_features`): `cong_{dep,arr,rwy,all}_{15,30,60}`, `queue_at_{pushback,takeoff}`,
  `sec_{since_prev,to_next}_dep_rwy`. These take the **unfiltered movement frame** as a separate traffic
  universe — arrivals compete for taxiways, so both phases count. Featurise ranking rows with ranking traffic,
  never with training traffic.
- **Time** (`add_time_features`): local minute-of-day, month, day-of-year, weekend flag, and sin/cos encodings.
  **Local** civil time, not UTC — an 07:00 local bank is a rush hour everywhere, but that is a different UTC
  hour at Istanbul than at Heathrow.
- **Route** (`add_route_features`): `planned_enroute_sec`, `is_longhaul`, `dest_region`, `dest_country`.
- **Stand** (`add_stand_features`): `stand_area`, `stand_full`, `stand_number`. Stand naming takes two
  incompatible forms — letter-prefixed at EDDF/EHAM/LFPG (`B41`, `V119`), purely numeric at
  EGLL/EDDM/LEBL/LIRF (`583`, `328`) where the leading digit is the terminal block. A single letter-prefix
  regex silently nulls 44.5% of rows. The grouping separates real terminals: EGLL T5 median 1,369 s vs
  T2 1,082 s, a 287 s spread inside one airport.
- **Wind** (`add_wind_components`): head/cross-wind against the runway heading parsed from `RUNWAY_mvt`.
  A raw wind direction cannot express this — 270° is a headwind on runway 27 and a tailwind on 09.
- **Weather regimes** (`add_weather_regimes`): `wx_lvp`, `wx_vis_very_low`, `wx_deice_regime`, `wx_snowing`,
  `wx_strong_gusts`, at the thresholds measured in `analysis.ipynb` §12.2.

### Leakage decisions, recorded

- **Excluded** (`EXCLUDE_FROM_MODEL`): `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` (target and its
  complement); `ARVT_3_flt` (flown arrival, after the event); and **`dq_implausible_target` /
  `dq_was_negative`, which are computed from the target** and would feed the answer straight into the fit.
  `dq_no_aircraft_type` is deliberately kept — it reads `AIRCRAFT_TYPE_mvt`, which is populated in the
  ranking set. Raw `datetime64` columns are dropped so a tree cannot memorise the training period.
- **`ARVT_1_flt` is used, narrowly, and this was verified.** This section previously listed it as a leakage
  trap. `planned_enroute_sec = ARVT_1_flt − EOBT_1_flt` is a *plan-side difference*: both endpoints come from
  the filed flight plan, so it is known before pushback. Checked rather than assumed — `ARVT_1_flt` differs
  from the flown `ARVT_3_flt` by a median of −280 s with IQR [−736, +114] and is exactly equal on only 0.07%
  of rows, so it is not a post-hoc copy of the actual. The signal is also predominantly between-route
  (r = 0.274) rather than within-route (r = 0.099), i.e. mostly haul length. The residual within-route part is
  the airline padding its own schedule, which is likewise known before departure. Using `ARVT_1_flt` as an
  *absolute* time, or using `ARVT_3_flt` at all, remains off-limits.

### Unseen-category exposure in the ranking set (measured)

Small but non-zero, so encoders need a defined fallback rather than emitting NaN:

| column | unseen categories | rows affected |
|---|---|---|
| `stand_full` | 24 | 339 (0.157%) |
| `AIRCRAFT_OPERATOR_flt` | 10 | 61 (0.029%) |
| `AIRCRAFT_TYPE_mvt` | 3 | 10 (0.005%) |
| `dest_country` | 2 | 15 (0.007%) |
| `RUNWAY_mvt` | 0 | 0 |

### Train/serve skew check (measured)

Medians and p90s of every new feature match between the training and ranking tables (largest median ratio
1.13 on `queue_at_takeoff`, which is a small-integer feature where a one-unit shift moves the ratio a lot;
p90s agree closely). No rescaling needed.

## 5. Validation protocol

### IMPLEMENTED — `src/validation.py`, run by `scripts/train_baseline.py`

**The split.** Train on **Jan–Jun 2025**; validate on **Dec 2025** (standing in for January) plus **Jul 2025**,
both across all ten airports. Both validation months lie strictly after the training window, so it is a genuine
forward backtest. Each (slice, airport) cell is re-weighted onto the share the leaderboard gives it, read from
`ranking.parquet` at run time — currently **44.3% January / 55.7% July across 20 scored cells**.

> ⚠️ **Re-measured 2026-09-07.** The organizers replaced `ranking.parquet` on 2026-09-04: 344,841 DEP rows
> instead of 215,876, and July now covers **all ten airports** rather than only EDDF/EGLL/EHAM. Every number in
> this section below was measured under the *old* 70/30, three-airport-July composition and needs re-running.
> `ranking_composition()` reads weights from the file so it self-corrects; `JULY_AIRPORTS` was hard-coded to
> three airports and has been fixed. Re-run `scripts/train_lgbm.py` for current numbers.

**The design tension, stated rather than hidden.** Mirroring the season and staying forward in time cannot both
be fully satisfied with one year of labels: the only January in the data is Jan-2025, at the very start, so
validating on it would mean training on months that follow it. December is the nearest held-out winter month
and is used as the January stand-in. Two costs to quote alongside any number: the model fits on six months
rather than twelve (understating the final all-data model), and December has atypical holiday traffic.

**Rows are never dropped from the validation set.** `trainable_mask` filters what the model *fits* on, not what
it is *scored* on. The validation slice retains its 110 anomalous rows — almost exactly the 108 in the real
ranking set, which is a good independent sign the protocol mirrors reality.

### Measured results — under the SUPERSEDED composition (70/30, three-airport July)

These stand as a record of the protocol working, not as current numbers. Re-measure before quoting.

| | weighted RMSE (full) | trimmed |
|---|---|---|
| per-airport median | 469.9 s | 386.0 s |
| raw `AOBT_3` proxy | 478.5 s | 396.6 s |
| **CatBoost, 88 features** | **339.3 s** | **221.5 s** |

**Decomposing the gap to the smoke test's 212.4 s** — and the answer is not what it first looked like. Running
the identical model and the identical naive Jan–Oct/Nov–Dec split, but scoring *every* validation row instead
of only the trainable ones:

| | naive split | mirrored split |
|---|---|---|
| trimmed (anomalous rows not scored) | 212.4 s | 221.5 s |
| full (anomalous rows scored) | 380.6 s | 339.3 s |

So the **composition mirroring costs only ~9 s** (212.4 → 221.5). The 127 s gap was almost entirely something
else: `scripts/smoke_test.py` applies `trainable_mask` to the whole frame *before* splitting, so its validation
set never contained the 156 anomalous Nov–Dec rows at all. It was reporting a trimmed number throughout.

Both effects are real and both belong in the protocol, but the ranking of them matters for where to spend
effort: **scoring the anomalous rows honestly is worth ~10× more than the composition weighting.** Note also
that the mirrored *full* number (339.3 s) is better than the naive full number (380.6 s) — Nov–Dec happens to
carry more anomalous rows than the mirrored slice does.

### Where the error actually is — one cell carries 59% of it

Decomposing the weighted MSE across the 13 scored cells:

| cell | RMSE | weight | share of weighted MSE |
|---|---|---|---|
| **jan LIRF** | **1,153.9 s** | 0.051 | **59.3%** |
| jul EGLL | 319.1 s | 0.095 | 8.4% |
| jan LFPG | 268.5 s | 0.084 | 5.3% |
| jan LTFM | 234.2 s | 0.104 | 4.9% |
| jan EGLL | 244.5 s | 0.089 | 4.6% |
| *(remaining eight cells)* | 162–224 s | — | 17.5% |

Inside that LIRF cell, **88 anomalous rows out of 12,068 (0.73%) carry 86.7% of its squared error**. LIRF
excluding them scores 448 s; including them, 1,223 s.

**Measured, not assumed: an explicit branch is worth 18.5 s.** 118 s is only the ceiling — what you would get
if those rows became perfectly predictable, which they will not be.

**The branch must key on `AIRCRAFT_TYPE_mvt` being null, not on `trainable_mask`.** The mask also catches
implausible targets, which requires the *label* and so is unavailable at prediction time. Of the 110 validation
rows failing the mask, only 89 have a null aircraft type; the other 21 are invisible to us when predicting.
Measuring the branch with the mask overstates it by 5 s:

| prediction for those rows | weighted RMSE | gain |
|---|---|---|
| current model (no branch) | 339.3 s | — |
| **honest branch: null aircraft type → training mean (6,763 s)** | **320.7 s** | **18.5 s** |
| oracle branch using `trainable_mask` (not applicable) | 315.5 s | 23.7 s |
| perfect prediction (ceiling) | 221.4 s | 117.9 s |

Three things follow. **The branch is worth doing** — 18.5 s for a handful of lines, at the top of the 15–20 s
originally estimated. **It is not already captured**: `dq_no_aircraft_type` is a model feature and CatBoost has
learned *something* from it, predicting a mean of 1,888 s for these rows against a normal ~1,000 s — but the
actual mean is 7,794 s, so it is barely a quarter of the way there. A tree fitted on `trainable_mask` rows
simply never sees a 7,000 s target and cannot extrapolate to one. **The training mean is a good enough
estimate** — the oracle mean scores 315.4 s against the training mean's 315.5 s, so nothing is lost to
estimating it from only 538 examples. The remaining 94 s between 315.5 s and the 221.4 s ceiling is
irreducible noise: those targets run to 87,605 s.

**Per-airport bias is small** (−48 s to +21 s across all 13 cells), so there is no systematic drift problem to
correct — the EDDF/EHAM 2026 drift is being absorbed by `proxy_taxi_aobt3` as predicted.


### Measured drift into 2026 — it is per-airport, not global

Comparing the *same month* year-over-year (median `proxy_taxi_aobt3`, which tracks the target closely):

| airport | Jan-2025 → Jan-2026 | Jul-2025 → Jul-2026 |
|---|---|---|
| EDDF | **+47 s** | **+33 s** |
| EHAM | **+50 s** | **+64 s** |
| LSZH | +12 s | — |
| LTFM | +10 s | — |
| EDDM, LEBL, LEMD, LFPG, LIRF | ±3 s | — |
| EGLL | −3 s | **−54 s** |

The aggregate proxy median looks like it jumped +59 s from training to ranking (958 s → 1,017 s), but that is
**mostly a mix effect**, not a global slowdown: the ranking set weights airports differently from the full
training year (July is EDDF/EGLL/EHAM only). Like-for-like, seven of ten airports are flat and the real drift
is concentrated at Frankfurt and Schiphol, with Heathrow moving the other way in July. Traffic volume is
**not** the cause — `cong_dep_60` and `queue_at_pushback` have identical Jan-2025 and Jan-2026 medians
(ratio 1.000).

Net effect on the target: implied Jan-2026 median ≈ 943 s against 906 s in Jan-2025, about **+4.1%**.

Two consequences for validation:
1. **Never judge a split on its aggregate distribution** — airport mix dominates it. Compare per airport.
2. A model keyed on `proxy_taxi_aobt3` absorbs most of this automatically, since a flight whose proxy is
   longer simply gets a longer prediction. The unobservable risk is the *residual* (proxy − target)
   relationship shifting at EDDF/EHAM, which no 2025 holdout can measure. Worth a per-airport bias audit on
   the final predictions.


- **No random k-fold as the primary validation** — the real evaluation is out-of-time (Jan/Jul 2026, unseen
  months). Use a time-based split, e.g. train on Jan–Oct 2025 / validate on Nov–Dec 2025, or a rolling-origin
  backtest across month boundaries.
- **Mirror the ranking set's composition, which is lopsided** (measured):
  - **January 2026 is 70% of the score** (152,719 of 215,876 rows); July 2026 is the other 30%.
  - **July 2026 contains only EDDF, EGLL and EHAM.** The other seven airports are scored on January alone.
  - So a validation slice built from Jan-2025 + Jul-2025 should apply that same 70/30 weighting and restrict the
    July portion to those three airports. A uniform year-long holdout scores a distribution we are not graded on.
- **Report RMSE both with and without the pathological tail.** The two numbers answer different questions: the
  full number is what the leaderboard shows, the trimmed number is what tells you whether a modelling change
  actually helped. A change worth 5 s of real RMSE is invisible under the noise of ~100 anomalous rows.
- Track RMSE (the actual contest metric) as the primary number for every model, not an afterthought beside MAE/R².
- Keep the covariate-shift findings in mind when picking the window — prefer a split that reproduces a divergence
  over one that looks artificially easy.

## 6. Modeling checklist

### LIRF investigation (2026-09-07) — mostly a negative result, and worth recording as one

Earlier notes described LIRF's non-anomalous rows as scoring "448 s against 160–320 s elsewhere" and treated
that as the largest remaining structured error. Both halves of that were wrong.

**1. The 448 s figure was not the model's.** It came from a constant-prediction baseline on the LIRF December
cell, not from the fitted model. Measured properly, LIRF's normal-row RMSE is **329 s**, against 263 s at EGLL
and 167 s at EHAM. Still the worst cell, but not the outlier it appeared to be — and its `rmse / target-sd`
ratio is 0.7 against 0.5–0.6 elsewhere, so most of the gap is simply that LIRF's taxi times vary more.

**2. It is not a LIRF problem.** LIRF's extra error is a long right tail of genuinely long taxis, and *every*
airport has the same structure. Rows with a target over 2,000 s are **2.29% of validation rows and carry 32.8%
of all squared error**; excluding them takes overall trimmed RMSE from 221 s to 183 s. LIRF simply has more of
them (3.0% of its rows against 0.4% at EHAM).

**3. The apparent "systematic under-prediction of long taxis" is a selection artifact.** Mean signed error on
rows with a target over 2,000 s is −523 s, which looks like a calibration failure. It is not: selecting rows
*by the outcome* produces exactly that pattern even from a perfect model. Binning by **prediction** instead —
the valid test — the model is calibrated to within ±1.8% in every decile, and to −0.9 s overall:

| predicted | n | mean predicted | mean actual | gap |
|---|---|---|---|---|
| ≤ 605 s | 23,018 | 498 s | 502 s | +0.8% |
| 940–1,194 s | 57,545 | 1,058 s | 1,056 s | −0.2% |
| 1,758–2,128 s | 4,604 | 1,893 s | 1,926 s | +1.8% |
| > 2,128 s | 2,302 | 2,613 s | 2,601 s | −0.5% |

**What this rules out.** There is no calibration correction, no LIRF-specific rule and no post-processing tweak
available here — the model is already predicting the conditional mean, which is exactly what minimises squared
error. The residual is genuine uncertainty given the features. `AOBT_3_flt` does partly see these events
(median proxy 1,624 s on long-taxi rows against 975 s on normal ones) but undershoots by ~715 s, and nothing
else in the feature set closes that gap. Any further gain has to come from a feature that genuinely
discriminates which flights will be slow, not from reweighting or re-calibrating what we have.


- [ ] **The bar is 373.0 s** — the per-airport median measured on trainable rows. Do not use the old
      "median 519.5 s vs AOBT_3 proxy 385 s" pair: those were computed on different row sets (the first
      including the pathological tail, the second excluding it), and on identical rows the median actually
      *beats* the raw proxy (373.0 s vs 380.9 s; 519.5 s vs 521.4 s over all rows). A per-airport bias
      correction on the proxy reaches 363.7 s. See `docs/findings.md` finding 2.
- [ ] Report the full-row number too, since that is what the leaderboard computes.

**Smoke test on the built feature set** (2026-09-06, CatBoost 500 iters, depth 8, all 88 features, plain
Jan–Oct / Nov–Dec 2025 time split — *not* the composition-mirroring protocol above, so treat it as a
feature-set sanity check and not as an expected leaderboard score):

| | RMSE on Nov–Dec 2025 |
|---|---|
| per-airport median | 377.9 s |
| raw `AOBT_3` proxy | 369.1 s |
| **CatBoost, 88 features** | **212.4 s** |

Top importances: `proxy_taxi_aobt3` (16.0), `proxy_taxi_eobt1` (15.1), `RUNWAY_mvt` (9.4), `stand_area` (9.2),
`ADEP_mvt` (7.7), `stand_full` (4.9). The stand/runway group carrying ~23% combined is the feature-engineering
payoff; the queue and congestion features rank low individually (`queue_at_takeoff` 0.96, `cong_rwy_15` 1.41).

Why 212 s is plausible rather than leakage: the model is effectively predicting the disagreement between two
independent off-block measurements (`AOBT_3_flt` from ADS-B vs the airport's `BLOCK_TIME_UTC_mvt`). That
disagreement has σ = 384 s but per-airport medians spanning −118 s to +296 s, so a large systematic component
is removable, and the 212 s residual is what is left after per-airport, per-stand and per-runway correction.
- [ ] **Branch the missing-`AIRCRAFT_TYPE_mvt` regime explicitly.** Those rows are a different distribution
      (mean 6,451 s / σ 11,170 s vs 987 s / 432 s), they are excluded from the main fit by `trainable_mask`, and
      108 of them are in the ranking set. Predicting their conditional mean instead of a normal taxi time is
      worth on the order of 15–20 s of final RMSE — more than most feature work will yield.
- [ ] Gradient boosting (xgboost/lightgbm/catboost, already installed) as the main candidate family; use native
      categorical support (lightgbm/catboost) where convenient instead of manual encoding.
- [ ] Use `optuna` for hyperparameter search, optimizing directly on the time-based validation RMSE from §5.
- [ ] Use `shap` to sanity-check feature importances against the EDA findings — features that dominate but make no
      operational sense are a leakage red flag, not a win.
- [ ] Record every model's validation RMSE plus which feature set/cleaning rules produced it, so the final model
      choice is traceable back to a documented experiment (needed for the reproducibility requirement anyway).

### The shipped model, exactly

One global **LightGBM regressor** — not an ensemble, not per-airport models. Everything airport- or
regime-specific is carried by features and by two post-processing rules, which measured better than splitting
the model (per-airport fits were never competitive: the airports share almost all of their structure, and the
small ones cannot support their own fit).

| | |
|---|---|
| library | `lightgbm` 4.7.0, `LGBMRegressor` |
| objective | `regression` (squared error — the contest metric is RMSE, so the conditional mean is what is wanted) |
| features | **127** — 112 numeric + 15 native categoricals |
| categoricals | `ADEP_mvt`, `ADES_mvt`, `RUNWAY_mvt`, `AIRCRAFT_TYPE_mvt`, `AIRCRAFT_TYPE_flt`, `AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`, `WK_TBL_CAT_flt`, `FLIGHT_RULE_mvt`, `FLIGHT_RULE_flt`, `FLIGHT_TYPE_flt`, `stand_full`, `stand_area`, `dest_region`, `dest_country` |
| training rows | 2,083,395 of 2,085,047 DEP rows (all 12 months), after `trainable_mask` |
| iterations | 1,400 fixed — no early stopping, because every held-out slice is already part of the validation protocol |

Hyperparameters (`data/processed/lgbm_best_params.json`, from the round-1 optuna sweep):

    learning_rate     0.0245929435      reg_lambda        19.3573292607
    num_leaves        464               reg_alpha          5.6135101845
    min_child_samples 14                max_bin          511
    colsample_bytree  0.4985357566      subsample          0.82801499 (freq 1)

> ⚠️ These were tuned against the **88-feature** set, and by a sweep that early-stopped on the slice it scored
> (see the warning in `scripts/tune_lgbm.py`). They are carried forward unchanged, so they are very likely not
> optimal for the current 127 features. Re-tuning — with the early stopping moved to an inner split — is the
> largest untried modelling lever.

**Two post-processing rules**, both in `src/postprocess.py`, applied in this order. Neither is learnable by the
regressor, because the rows they handle are excluded from the fit by `trainable_mask`:

1. `apply_regime_model` — rows with a null `AIRCRAFT_TYPE_mvt` (388 in the ranking set) are predicted by a
   **second, tiny LightGBM** fitted on the 1,507 such training rows, regressing `proxy_taxi_sched - taxi`
   rather than `taxi`. Worth 8.6 s.
2. `apply_extreme_proxy_override` — rows whose `proxy_taxi_aobt3` exceeds 7,200 s *and* is corroborated by at
   least two plan-side off-block estimates (2 rows in the ranking set) take the midpoint of proxy and plan
   consensus. Worth 36.1 s.

Then a floor at the 0.1st percentile of trainable targets (119 s), which raised 79 predictions in v3.

Reproduce end to end:

    python scripts/build_features.py          # base feature tables
    python scripts/build_surface_features.py  # rolling surface state
    python scripts/train_lgbm.py --tuned      # the validation protocol number
    python scripts/build_submission.py --team diligent-fireplace --version <N>

## 7. Submission checklist

### IMPLEMENTED — `scripts/build_submission.py`

    python scripts/build_submission.py --team <team-name> --version <N>

Fits on **all 12 months** (2,085,047 DEP rows), unlike `train_baseline.py` which fits on Jan–Jun so its holdout
can stay forward in time. The validated 320.7 s is therefore a *pessimistic* estimate of this model.

Two post-processing rules, applied after the regressor and documented because neither is learnable by it:

1. **Missing-aircraft-type branch** — null `AIRCRAFT_TYPE_mvt` → the training conditional mean. Worth 18.5 s.
2. **Prediction floor** at the 0.1st percentile of observed trainable targets. RMSE is unbounded below but
   taxi time is not.

The script runs every check below and refuses to write the file if any fails.

### Leaderboard context (observed 2026-09-07)

Top scores cluster at **263.5–290** across the visible top 50. Two things worth carrying into planning:

- **Iteration is not what wins.** `enthusiastic-daisy` has filed **566 submissions** and sits 4th at 269.87;
  `youthful-giraffe` leads at 263.46 on its **7th**. That is a problem near its noise floor, where gains come
  from structural insight rather than search — so prefer regime handling and feature work over long sweeps.
- **A competent first submission scores ~289** (`vibrant-tornado_v1` at 288.87), so the whole contest is fought
  in the ~25 s above that. The 18.5 s regime branch is comparable to the gap between 1st and ~20th.

**RESOLVED — `usedPairs` = 344,841 is the size of the truth set**, and the local `submitting.parquet` was
simply stale. The organizers replaced `ranking.parquet` / `submitting.parquet` on **2026-09-04**; the older
copy had 215,876 DEP rows. Submission v1 was built from the stale file and **rejected**:

    "status": "Failed",
    "error": "Submission is missing 128965 required rows present in truth. (No zero-fill.)"

215,876 + 128,965 = 344,841. **Lesson: check the bucket against local before every submission** —
`mc ls prc/prc-2026-datasets/` and compare sizes and timestamps. The training files were unchanged.

### Submission log

| version | model | branch | validated | **leaderboard** | notes |
|---|---|---|---|---|---|
| v1 | CatBoost, 88 feat | constant mean | 320.7 s | **REJECTED** | built from the stale pre-2026-09-04 ranking file |
| v2 | LightGBM tuned | linear on `proxy_taxi_sched` | ~332 s | **329.76 s** | first valid score |
| v3 | LightGBM tuned, 127 feat | delta regime model + extreme-proxy override | 281.5 s | **287.10 s** | see §8. The 2-row bet paid: a failed override would have landed near 348 |
| v4 | LightGBM r3 params, 139 feat, `cat_smooth=100`, 3-seed bag | unchanged from v3 | 276.50 s | **282.23 s** | see §11. Runway configuration + stand rotation were the substance |
| v5 | + 17 off-block gaps + 4 OSM geometry (160 feat) | unchanged from v3 | 275.27 s | **279.73 s** | two sub-noise families that were additive together; beat its own prediction by 1.3 s |

**The protocol's error is a stable OFFSET, not a band — established 2026-09-26 by v4.**

| submission | predicted | actual | error |
|---|---|---|---|
| v2 | ~332 s | 329.76 s | −2.24 s (pessimistic) |
| v3 | 281.49 s | 287.10 s | **+5.61 s** |
| v4 | 276.50 s | 282.23 s | **+5.73 s** |
| v5 | 275.27 s | 279.73 s | **+4.46 s** |

v3 and v4 agreed to within 0.12 s, which looked like a constant. **v5 then came in 1.27 s off that**, so the
honest statement is an offset of **+5.3 ± 0.7 s** — much tighter than the old ±6 s band and usable as a
correction, but not the constant two points suggested. Three points is still few; expect it to move again. v2 sits apart because it was a
different model family — 88 features and the old linear regime branch — and its "~332" was itself a loose
estimate rather than a protocol run.

What this does *not* license: ranking two changes that differ by less than about a second, since the offset is
calibrated on two points and the per-change measurement noise is unchanged. It does mean an offline target can
be set directly — the leader's 224 s corresponds to **offline ~219 s**, against v5's 275.27 s — a gap of **~56 s offline**
with no identified path (see §11). The most likely source of the drift is the override rows: the two it fires on in the ranking set are
not the two it fires on in validation, and they need not be equally well recovered. That is far better than the protocol's known compromises (December standing
in for January, a six-month fit, a 1.6× higher anomaly rate than the ranking set) had any right to deliver, and
it means offline measurements can now be trusted to rank changes without spending a submission on each.

Leaderboard context at the time of scoring: leaders at 263.5 s, a competent first entry ~289 s. So v2 sits
**~66 s behind the lead**. Of that gap, at most ~40 s is available from better handling of the anomalous rows
(the validated ceiling for perfect anomaly prediction is 295.1 s against v2's 335.7 s under identical
conditions). The remaining ~26 s has to come from the normal-row model, where our trimmed RMSE is 242.6 s.

### Filing procedure (confirmed working 2026-09-07)

The web console is **read-only for teams by design** — its object browser shows `Access: R` and *Upload File*
is disabled. That is not a permissions bug; uploads go through the MinIO client:

    brew install minio-mc          # NOT `brew install minio` (server) or `mc` (Midnight Commander)
    # MinIO console -> Access Keys -> Create   (secret shown once)
    mc alias set prc https://s3.opensky-network.org/ ACCESS_KEY SECRET_KEY
    ./scripts/upload_submission.sh submissions/diligent-fireplace_v<N>.parquet

A `<filename>_result.json` appears in the same bucket within ~1 minute carrying either the score or a failure
reason. Fetch it with `mc cp prc/prc-2026-diligent-fireplace/<file>_result.json data/results/`.



- [ ] Predict only for `ranking.parquet` rows where `PHASE_mvt == "DEP"`.
- [ ] Output columns exactly `MVT_ID_mvt`, `TAXITIME_SEC_mvt`, matching `submitting.parquet`'s dtypes and row order/
      ID set exactly (`set(MVT_ID_mvt) == dep_ids` check, as already scaffolded in the notebook's sanity-check cell).
- [ ] No NaNs, no negative predictions (clip at a sane floor, e.g. a few seconds, if a model predicts negative).
- [ ] Re-run the full pipeline end-to-end from raw parquet files once before final submission, to catch any
      notebook-state drift (variables reused out of order, stale intermediate results).

## 8. Round 2 (2026-09-13) — error budget, and what it says to work on

### The budget, measured

Decomposing the validated weighted MSE of the v2-equivalent model (330.9 s) by row population.
This is the number that should drive where effort goes, and it was not obvious in advance:

| population | rows (of 356,390) | share of weighted MSE |
|---|---|---|
| normal | 355,919 | 53.5% |
| missing `AIRCRAFT_TYPE_mvt` | 428 | 23.4% |
| broken label, **no visible signature** | 43 | 23.1% |

Two populations of ~0.1% of rows carry **47%** of the metric. Oracle predictions give 289.6 s
(no-aircraft-type solved) and 290.2 s (invisible solved); both solved, 242.1 s.

### The invisible rows: the bulk is unreachable, the extremes are not

Worth stating carefully, because the two halves point in opposite directions.

**Detecting them as a population does not work.** `proxy_taxi_sched > 7200` captures **96%** of
them, but that band is 1.9% of all rows and only 0.35% pure. Under a mixture prediction the
gain scales as `q²Δ²/s` (q = base rate, s = band size, Δ = mean separation), giving **~0.15 s**
against a 41 s oracle. Confirmed empirically: widening `trainable_mask` to include them cost
4 s trimmed for 0.4 s full. Do not spend more time on population-level detection.

**But squared error is not carried by the population — it is carried by its extremes.** Of the
43 invisible validation rows, the two with corroborated implausible proxies carry most of the
group's weighted MSE, and the override below recovers **36 s** from exactly those two. The
lesson generalises: under RMSE, look for the handful of rows with the largest recoverable
errors before trying to model the population they belong to.

### What worked

| change | validated weighted RMSE | Δ |
|---|---|---|
| v2 equivalent (88 features, linear branch) | 331.1 | — |
| + rolling surface-state features | 328.1 | −3.0 |
| + off-block corroboration features | 326.2 | −1.9 |
| + delta-target regime model | 317.6 | −8.6 |
| + corroborated-extreme-proxy override | **281.5** | −36.1 |

Trimmed (normal rows only): 242.8 → **237.3**. Filed as **v3** and **scored 287.10** against v2's 329.76, a
**42.7 s** gain on the board. The override's 2-row bet paid — had it failed, the score would have come back near
348 rather than within 6 s of the prediction.

**Surface state** (`src/surface_state.py`) — backward-looking rolling means of *realised*
durations, not counts: recent arrival taxi-in times (ARR rows are 100% labelled in
`ranking.parquet`, and this was previously unused), recent departure `MVT_TIME - AOBT_3`, and
recent `AOBT_3 - EOBT_1` delay. Windows are half-open and backward-only so a flight can never
see its own outcome through its neighbours. Train/serve skew checked: coverage differs by at
most 0.8%, medians by 0.94-1.04x.

**Corroboration** (`src/cleaning.py`) — `offblock_agree_n` and friends: how many of the four
plan-side off-block estimates sit within 30 min of `AOBT_3_flt`. Where all four agree the
proxy lands within 15 min of the target 86.3% of the time; where none do, 60.2%.

**Delta regime model** (`fit_regime_model`) — fits the missing-aircraft-type rows on
`proxy_taxi_sched - y` instead of `y`. Beats the shipped linear on 5 of 6 disjoint month
splits; the exception is an 89-row December holdout, and July (which dominates this regime's
scored exposure 4:1) is where it wins most.

**The extreme-proxy override** (`apply_extreme_proxy_override`) — the single largest item, and
the one to understand before trusting the headline. A tree cannot predict outside the target
range it was fitted on, so a genuinely 24-hour taxi is predicted at ~1,000 s. Usually that is
correct, because such a proxy is normally just a bad `AOBT_3_flt`. But when the scheduled,
initial, last and estimated off-block times *all* sit at the same implausible offset, the
record's date is shifted as a whole and the proxy is right.

> ⚠️ **This rests on 2 rows of 344,841.** It is a real bet, not a free win. What justifies it:
> the rule separates the 19 training rows with `proxy > 7200` perfectly — fires on 7, all with
> genuinely large targets; abstains on 12, all ordinary taxis of 877-6,591 s. It was
> re-derived from **Jan-Jun alone** (6/6 correct) and is then **3/3 correct on the held-out
> Jul+Dec window**, so the 36 s is out-of-sample, not circular. Downside if the two ranking
> rows are not shifted records: roughly symmetric, about +31 s.

### What did not work — recorded so it is not re-litigated

- **Boosting from the proxy** (`init_score = proxy_taxi_aobt3`) looked like a 34.5 s win
  (328.0 → 293.5). It is not: **one row** supplied it, and dropping that row made the variant
  *worse* than the baseline (293.5 vs 293.2). It also costs ~1 s on the bulk (trimmed 238.3 →
  239.4). The override captures the same upside without the bulk cost. **Always decompose a
  tail gain by row before believing it.**
- **Finer-grained surface features** (per runway, per apron, per apron→runway): 238.3 → 238.3
  trimmed, 328.0 → 328.1 full. Dead flat. The airport-level state plus the existing
  stand/runway categoricals already carry it. They remain in the build and are harmless, but
  could be pruned.
- **`stand_area x RUNWAY_mvt` as a categorical**: no change. LightGBM already recovers it.
- **Two-component mixture on the regime** (`|SCHED - BLOCK| <= 60` splits it 48/52): worse on
  all 6 splits, and the *mode-oracle* is worse than the plain linear (7,302 vs 4,180). The
  bimodality is real but does not align with squared error.
- **Arrival-side clock gap** (`ARVT_3_flt - MVT_TIME_UTC_mvt` on ARR rows) as a proxy for the
  unobservable departure offset `AOBT_3 - BLOCK_TIME`: σ = 57 s, within-airport correlation
  −0.04, per-airport correlations flipping sign. No shared drift component.
- **Duplicate matched records** for the unmatched flights: only 3.0% have a same-airport DEP
  twin within 6 h, and their block times never agree.
- **Exact identities** between `BLOCK_TIME_UTC_mvt` and any ranking-available timestamp: the
  best is `AOBT_3_flt` at 0.65% exact, and nothing identifies those rows in advance.
- **Extending the override to `[3600, 7200)`** (904 ranking rows, 180x the exposure): the
  group mean beats the proxy there (1,301 s vs 1,844 s). The override is correctly confined to
  the extreme band.

### Ensemble — measured, and not worth it

CatBoost on the identical 127-feature split scores **289.9 s** full / 246.9 s trimmed, i.e.
worse than LightGBM's 281.5 / 237.3 alone. Blending helps, but barely:

| weight on LightGBM | full | trimmed |
|---|---|---|
| 0.00 (CatBoost only) | 289.9 | 246.9 |
| 0.60 | 281.0 | 236.6 |
| **0.75 (best)** | **280.6** | **236.1** |
| 1.00 (LightGBM only) | 281.5 | 237.3 |

0.9 s, with the weight chosen from a 7-point grid on the same holdout it is scored on — so the
honest expectation is less. Against a ~20 min CatBoost fit on half the data (~40 min on all of
it) this does not earn its place in the submission path. Revisit only if the two models are
made to disagree more, e.g. by giving CatBoost a different feature subset.

### Where the remaining error is (after round 2)

| oracle on | resulting RMSE | headroom |
|---|---|---|
| normal (355,919 rows) | 152.3 s | 129.2 s |
| missing aircraft type (428 rows) | 243.2 s | **38.3 s** |
| invisible (43 rows) | 275.9 s | 5.6 s |

The invisible population is close to exhausted — the override took its headroom from 41 s to
5.6 s. **`jul LIRF` alone carries 35.8% of the remaining weighted MSE** (RMSE 785 s on 15,211
rows), and it is the cell where the missing-aircraft-type regime concentrates. That regime is
the largest identifiable target left, and note it is fitted on only 538 rows under the mirrored
protocol against 1,507 in the shipped model — so the real model is better there than 281.5
implies.

## 9. Publication checklist (GPLv3 / reproducibility)

The challenge requires the source published on GitHub under **GPLv3**, documentation sufficient for someone
else to reproduce the results, and every supplementary dataset openly licensed and documented.

Done:

- [x] `LICENSE` — verbatim GNU GPL v3.0.
- [x] `README.md` — task, result, quickstart, method, validation design, repo map, data licences.
- [x] `requirements.txt` — pinned to the versions the published results were produced with (Python 3.14.5),
      with the macOS `brew install libomp` prerequisite called out.
- [x] `docs/features.md` — all 127 model inputs with measured importance and coverage.
- [x] `docs/data_dictionary.md` — all 31 delivered columns.
- [x] External data documented and open: Open-Meteo (CC BY 4.0), `holidays` (MIT), OurAirports (public
      domain), each with a `.meta.json` provenance sidecar or an in-file provenance note.
- [x] Open-Meteo attribution in `README.md` — **required** by CC BY 4.0, not optional.
- [x] Rejected sources logged with reasons (§1), so the licence reasoning is auditable.
- [x] Secrets excluded — `credentials.json`, `*credentials*.json`, `.env` gitignored.
- [x] `catboost_info/` (training logs) gitignored.
- [x] Experiment scripts made portable — they cached to an absolute path containing a local username and a
      session id; they now use `experiments.harness.CACHE` under the gitignored `data/processed/`.

Remaining, and each needs a deliberate decision:

- [x] **MinIO access key rotated** (2026-09-13, by the user). `credentials.json` at the repo root now
      holds the *superseded* key and can be deleted; `mc` uses `~/.mc/config.json` and the pipeline never
      reads it. It stays gitignored either way.
- [ ] **First commit + push.** The repo exists (github.com/saurabhdamle11/prc-2026, cloned at `prc-2026/`,
      containing only GPLv3 `LICENSE`); publishable files were copied in on 2026-10-14, **uncommitted**.
      A scan of every file to be committed found none of the MinIO key values, no local paths, VM address or
      e-mail. Committing and pushing still need an explicit go-ahead.
- [x] **Tuned hyperparameters published** (2026-10-14). `build_submission.py` reads
      `data/processed/lgbm_best_params_r3.json`, which the blanket `data/processed/` ignore had hidden — a
      reproducer would have needed the 60-trial sweep. `.gitignore` now re-includes both params files and
      `lgbm_study.csv`.
- [x] **OpenStreetMap (ODbL 1.0) and IEM METAR (public domain) documented** in `README.md` with
      attribution; the OSM extracts are committed. OpenStreetMap's ODbL attribution is mandatory.
- [x] **OpenSky ADS-B pilot data excluded** (`data/external/adsb_ground/` gitignored): unused by the model and
      not ours to redistribute.
- [x] **Final phase documented**: README covers the four-month scoring file, `ranking_path()`, and the
      `_final` filename.
- [x] **`analysis.ipynb` outputs cleared** (2026-09-13): 56 outputs and 41 execution counts stripped,
      1,307,835 -> 55,737 bytes, no base64 PNGs remaining. The pre-strip copy is kept at
      `.backups/analysis_with_outputs.ipynb` (gitignored); re-run the notebook to regenerate figures.
      Re-strip before every commit with `python scripts/clear_notebook_outputs.py`, or gate it in CI /
      a pre-commit hook with `--check`, which exits non-zero if any output has crept back.
- [ ] Optional: per-file GPLv3 headers. A `LICENSE` plus the `README.md` statement is the usual minimum and
      is what is in place; the FSF also recommends a short notice at the top of each source file.

## 10. Round 3 (2026-09-19) — the hyperparameter sweep, corrected

`SKILLS.md` §6 flagged re-tuning as "the largest untried modelling lever". Before running a
sweep it was worth reading `scripts/tune_lgbm.py` rather than trusting its own warning, and it
turned out to have **three** defects, not the one it documented.

### What was wrong with the round-1 sweep

1. **It never merged `train_surface.parquet`.** `tune_lgbm.py` loaded `train_features.parquet`
   alone, so `model_features()` returned **93** columns where the shipped model uses **127** —
   all 34 rolling surface-state features were invisible to it. The parameters in
   `lgbm_best_params.json` were therefore tuned for a model that is not the one being
   submitted. This is the defect that actually matters, and it was not in the docstring.
2. **It early-stopped on the slice it scored** (`eval_set=[(val, yv)]`), so each trial's
   iteration count was chosen from the rows the trial was graded on. Now stopped on an
   **inner split** — the last training month (month 6, 182,943 rows) — held out of the fit,
   with the whole training window refitted at the chosen count afterwards. That second fit is
   what makes the scored model the one that ships; `--no-refit` skips it and halves the cost,
   at the price of scoring a model that has seen 5 of 6 training months.
3. **It applied the superseded v2 regime branch** (a constant conditional mean) instead of the
   shipped delta regime model plus extreme-proxy override.

Defect 3 is **ranking-neutral** and worth understanding rather than just fixing: both
post-processing rules overwrite their rows with values that do not depend on the sweep's
hyperparameters at all — `_REGIME_PARAMS` is a fixed dict and the override is rule-based — so
they add a *constant* to the weighted MSE, and RMSE is monotone in MSE. It changes the number
printed, not which trial wins. Fixed so the output is directly comparable to the protocol's
281.5 s.

### The search space was also too narrow

Three of the round-1 winners sat on a boundary of the space that produced them, which is the
usual sign that the optimum is outside it:

| parameter | round-1 best | round-1 range | round-3 range |
|---|---|---|---|
| `learning_rate` | 0.0246 | 0.02–0.15 | **0.015**–0.15 |
| `min_child_samples` | 14 | 10–300 | **5**–300 |
| `max_bin` | 511 | {127, 255, 511} | {127, 255, 511, **1023**} |
| `num_leaves` | 464 | 63–1023 | 63–**1500** |
| `colsample_bytree` | 0.499 | 0.4–1.0 | **0.3**–1.0 |
| `reg_lambda` / `reg_alpha` | 19.4 / 5.6 | ≤50 / ≤10 | ≤**100** / ≤**20** |

### The harness reproduces what shipped — verified

The sweep is seeded with the incumbent parameters as **trial 0** (`INCUMBENT`, plus
`study.enqueue_trial`), so every run measures the current model under the corrected protocol
before exploring. It returned **281.49 s full / 237.30 s trimmed**, matching round 2's
documented 281.5 / 237.3 exactly. The scaffolding is therefore faithful, and any movement is
attributable to the parameters rather than to the rewrite.

Incidentally, inner-split early stopping chose **1,118** iterations for the incumbent, against
the **1,400** fixed rounds the submission uses and the 1,150 default in `train_lgbm.py` — so
the shipped iteration count was already close to right.

### Cost

~4.5 min per trial with refit on (380 s for the incumbent's 1,118 iterations, 150 s for a
306-iteration trial — time scales with the chosen iteration count, so low learning rates are
the expensive ones). A 60-trial sweep is roughly 4.5 h.

> The sweep writes to `data/processed/lgbm_best_params_r3.json`, **not** to
> `lgbm_best_params.json`. Promote it only after confirming the gain through
> `scripts/train_lgbm.py --tuned`, and remember the protocol is good to ±6 s (§7) — a 2 s
> improvement chosen from 60 trials against one holdout is not a result.

> ⚠️ **LEADERBOARD MOVED — the leader is at 224 s, not 263.5 s** (observed 2026-09-26; the
> 263.5 s figure in §7 dates from 2026-09-07). This is not a quantitative update, it invalidates
> a strategic assumption that rounds 2-4 were all built on.
>
> **v4's *trimmed* weighted RMSE is 231.49 offline = 237.2 on the board.** That is the score we
> would get with **every anomalous row predicted perfectly** — and it is still **13.2 s worse
> than the leader**. The tail is not where the competition is being won and cannot be; the
> arithmetic forecloses it regardless of how good the regime model gets.
>
> To reach 224, the **normal-row** model must improve by 13 s (leader solves anomalies
> perfectly) to 41 s (leader halves them). Round 2 measured the normal-row oracle at 152.3 s,
> so this is well inside what the data permits — which means the leaders have information we
> do not, not merely a better-tuned model.
>
> The prime suspect is a **self-computed off-block time from raw ADS-B**: `AOBT_3_flt` is the
> first detected *movement* while the target is measured from *off-block*, and §11 measured
> that gap at **+2,758 s** on taxis over an hour — precisely the 4.2% of rows carrying 30.1%
> of the weighted MSE. This challenge is run by OpenSky and grants participants data access.
> See `docs/adsb_offblock_plan.md`, which now carries the query spec and a one-airport-month
> kill-switch.

## 11. Round 4 (2026-09-26) — the budget re-measured, and three directions closed

Round 3's sweep is finished and its verdict is **no**: 60 trials moved the mirrored protocol
from the incumbent's 281.49 s / 237.30 s to **279.82 s / 235.62 s**, a 1.7 s gain chosen as
the best of 60 against one holdout, against a protocol good to ±6 s (§7). The top twelve
trials span **0.65 s** — the objective is flat in hyperparameter space. `lgbm_best_params_r3.json`
is kept and is a marginal improvement worth carrying, but **hyperparameter search is an
exhausted lever and should not be run again.**

### The budget has moved — `experiments/e19_budget_r4.py`

Round 2's budget was measured against the v2-equivalent model. Re-measured against the
current one (r3 parameters, 127 features, both post-processing rules), the shares have shifted
because the normal-row model improved and the anomalous populations did not:

| population | rows (of 356,390) | share of weighted MSE | round-2 share |
|---|---|---|---|
| normal | 355,919 | **70.6%** | 53.5% |
| missing `AIRCRAFT_TYPE_mvt` | 428 | **25.6%** | 23.4% |
| invisible broken label | 43 | **3.8%** | 23.1% |

**One cell is 35.6% of the whole metric: `jul LIRF`** (RMSE 779.5 s on 15,211 rows, against a
150–300 s norm elsewhere; the next-largest cell is `jul LTFM` at 7.4%). Inside it, 337
missing-aircraft-type rows carry **64%** of the cell.

Rome in July is genuinely slow and it is **seasonal, not a 2025 artifact** — LIRF's share of
taxis over 1,800 s runs smoothly from 4.1% in January to **13.9% in July**, mean taxi peaks in
July at 1,287 s against ~1,050 s in winter, and the slowdown is uniform across every stand area
and runway rather than a mix effect. July 2026 will look the same, so the protocol is not
being pessimistic here.

### The mechanism behind the long-taxi error — `experiments/e22_calibration.py`

`proxy_taxi_aobt3`, the highest-gain feature in the model, **inverts in the tail**:

| true taxi | actual | `px_aobt3` | `px_eobt1` | `AOBT_3 − BLOCK` (the hold) |
|---|---|---|---|---|
| [0, 1200) | 833 | 873 | 1,144 | −67 |
| [1200, 1800) | 1,381 | 1,207 | 1,741 | +122 |
| [1800, 2700) | 2,008 | 1,435 | 2,401 | +478 |
| [2700, 3600) | 2,995 | **1,199** | 3,183 | +1,376 |
| [3600, ∞) | 4,312 | **1,079** | 4,014 | **+2,758** |

`AOBT_3_flt` is the first ADS-B *movement*; the target is measured from *off-block*. An
aircraft that pushes back and then holds on the apron has its entire hold skipped by the ADS-B
proxy, so the best feature in the model points the wrong way exactly where the error is.
`proxy_taxi_eobt1` tracks the target through the tail because a filed estimate cannot miss a
hold it never observed. Normal rows with a true taxi ≥ 1,800 s are **4.2% of rows and 30.1%
of the weighted MSE**.

### v4, measured end to end — `experiments/e30_final_r4.py`

Every surviving change measured *together* rather than summed, because component gains do not
have to add:

| | full | trimmed |
|---|---|---|
| v3 as shipped | 281.49 | 237.30 |
| **v4 candidate** | **276.50** | **231.49** |
| delta | **−4.99** | **−5.81** |

Configuration: 139 features (127 + 12 configuration/rotation), `lgbm_best_params_r3.json` at
3,546 iterations, `cat_smooth=100`, 3-seed bag, then the unchanged delta regime model,
extreme-proxy override and floor.

The raw regressor scores **530.06** full / 231.49 trimmed before post-processing, and the two
rules pull it to 276.50 while leaving trimmed *identical*. That is the check that they only
move the rows they are meant to: any drift in the trimmed number between the two lines would
mean a rule had reached into the normal population.

**Deliberately excluded: the 17 off-block gap features** (`e23`), at −0.52 s. Wiring them into
production means touching `src/cleaning.py` and both feature builds for a change a tenth the
size of the protocol's precision. Everything shipped in v4 is individually worth more than a
second and was measured on both the full and the trimmed number.

### The categorical parameters — `experiments/e29_categorical_params.py`

Rounds 1 and 3 swept `learning_rate`, `num_leaves`, `min_child_samples`, `colsample_bytree`,
`subsample`, `reg_lambda`, `reg_alpha` and `max_bin`. **None of those affects how LightGBM
splits a categorical feature**, and categoricals are roughly a third of this model's split
gain (`stand_area` 11.6%, `stand_full` 6.5%, `RUNWAY_mvt` 5.6%, `AIRCRAFT_OPERATOR_flt` 3.4%,
`ADES_mvt` 3.2%). Four parameters govern that path and all four were still at their defaults.
One-factor-at-a-time around the shipped configuration, deliberately not a joint search:

| change | full | trimmed |
|---|---|---|
| defaults | 279.82 | 235.62 |
| `max_cat_threshold` 64 / 128 / 512 | 280.55 / 280.60 / 280.68 | 236.37 / 236.28 / 236.52 |
| `cat_smooth` 1.0 | 280.47 | 236.23 |
| **`cat_smooth` 50.0** | **279.35** | **235.02** |
| `cat_l2` 1.0 / 50.0 | 279.85 / 280.36 | 235.63 / 236.13 |
| `min_data_per_group` 20 / 500 | 280.11 / 280.23 | 235.89 / 236.05 |

Two things worth keeping. **`max_cat_threshold` is not the binding constraint it looked like**
— `stand_full` has 2,723 levels against a default of 32, which seemed obviously too tight, but
every loosening made the model worse. The default is doing useful regularisation, not
truncating signal. **`cat_smooth` is monotone over the grid** (1.0 → 10 → 50 improves
steadily), which is what distinguishes it from a lucky draw; more shrinkage of each category's
gradient statistic toward the parent is the right call when 2,723 stands are spread over a
million rows and the rare ones are mostly noise.

> Timing note: the `cat_l2=1.0` fit took **1,671 s** against ~150 s for its neighbours, with
> nothing in its parameters to explain it. The round-3 study shows the same thing — trials 51
> and 54 took 2,816 s and 3,210 s against a ~250 s median, and its wall clock was ~14 h against
> 7.1 h of summed trial time. The cause is not established. Budget wall-clock for sweeps on
> this machine at roughly **2x** the sum of expected fit times.

### Sizing the ADS-B off-block prize BEFORE buying it — `e31`, `e32`, `e33`

With the leader at 224 s and the tail arithmetically incapable of closing the gap, the only
candidate left is a self-computed off-block time. Rather than commit the remaining window to it
on a hunch, it was **simulated first**, using data already in hand: training rows carry
`BLOCK_TIME_UTC_mvt`, so a detector of any quality can be faked as
`MVT_TIME - (BLOCK_TIME + noise)`, hidden on a fraction of rows to mimic coverage, and fed to
the real model as one extra feature.

**The bar (`e31`).** Every existing off-block estimate, scored against the truth on the
protocol's weighting: `AOBT_3_flt` **422.9 s** (median bias only −28 s but p90 abs error 536 s
— unbiased and heavy-tailed, the signature of missed holds), `EOBT_1_flt` 753 s, `LOBT`/`IOBT`
825 s, `SCHED` 2,588 s. The v4 model reaches **231.5 s** by combining them, i.e. it already
halves its best single input.

**The prize (`e32`).** Implied board score, using the measured +5.73 s protocol offset:

| simulated detector | full | trimmed | implied board |
|---|---|---|---|
| none (v4) | 276.50 | 231.49 | 282.23 |
| RMS 60 s, 90% coverage | 134.30 | 98.47 | **140.03** |
| RMS 150 s, 90% coverage | 159.79 | 132.89 | **165.52** |
| RMS 300 s, 90% coverage | 195.71 | 166.44 | **201.44** |
| RMS 150 s, 70% coverage | 196.29 | 162.76 | **202.02** |
| RMS 150 s, 98% coverage | 146.38 | 118.54 | **152.11** |

Accuracy and coverage trade off almost symmetrically — neither is a cliff.

**The robustness check (`e33`).** `e32` assumes Gaussian error independent of the target, which
is exactly the assumption `AOBT_3_flt` violates. Re-run with **heteroscedastic** error — fine on
ordinary pushbacks, bad on the holds:

| detector | implied board |
|---|---|
| flat 150 s | 166.30 |
| 150 s normal / **800 s on holds** | 187.26 |
| 150 s normal / **2,000 s on holds** | 209.02 |
| 150 s / 2,000 s on holds, **70% coverage** | **222.34** |

**Every scenario beats 224**, including one where the detector fails on holds nearly as badly as
`AOBT_3_flt` does *and* misses 30% of flights. The margin in that last case is only 1.7 s, so the
claim is not "cannot fail" but "failure requires being worse than a harsh worst case". The
load-bearing assumption is **150 s on ordinary pushbacks** — a 2.5-minute window on ground ADS-B,
achievable wherever coverage exists.

> These are **sizing simulations, not models**. They use the target's other half and could never
> ship. Their only purpose was to decide whether to start the project — and the answer is an
> unambiguous yes: everything rounds 2-4 produced was worth 4.87 s combined; the pessimistic end
> of this is worth **60 s**.

### The embedding MLP — the one untried model class, worth ~1.25 s in a blend

Torch MLP with learned embeddings for every categorical (`stand_full`'s 2,723 levels get a
64-dim vector — the thing trees cannot do), 1.08M parameters, trained on the same 160 features
and the same mirrored split. Run on the Oracle instance, not the laptop.

| model | standalone full / trimmed | residual corr. with LightGBM | best blend |
|---|---|---|---|
| LightGBM v5 (3-seed) | **273.72 / 230.21** | — | — |
| `e39` MLP, fixed 15 epochs | 284.20 / 242.74 | **0.9003** | **272.50** (−1.22 s) |
| `e41` MLP, early-stopped (8 ep), 3-seed | **280.42 / 238.06** | 0.9307 | 272.76 (−0.96 s) |
| `e42` three-way blend | — | — | 272.47 (−1.25 s) |

**The most useful finding here is counterintuitive: fixing the overfitting made the blend
worse.** Early stopping improved the MLP by 3.8 s standalone and *raised* its correlation with
LightGBM from 0.9003 to 0.9307 — it converged toward the same solution. For a model whose only
job is to diversify an ensemble, tuning it for standalone accuracy is actively counterproductive.
Worth remembering before "improving" any future ensemble member.

The three-way blend adds **0.03 s** over the best two-way, i.e. nothing: once one MLP is in,
the second is redundant.

**Verdict: not shipped.** The gain is ~1.25 s, which sits inside the 1.55 s platform spread
(below), the weights are chosen on the slice they are scored on, and shipping it means adding
MLP training to the submission path — a whole extra model class in the reproducibility burden
for about a second. Revisit only if something else closes the real gap first.

### ⚠️ Results are platform-dependent by ~1.5 s — thread count matters

The identical v5 configuration — same code, parameters, seeds, data, LightGBM 4.7.0 — scored
**275.27** on the laptop (10 threads) and **273.72** on the Oracle instance (4 threads).
LightGBM's histogram accumulation is order-dependent in floating point, and over 3,546
iterations x 3 seeds that diverges by **1.55 s**.

Consequences, and they are not small:

* **Several "gains" recorded this round are inside that spread** — the off-block gaps
  (−0.52 s), the OSM geometry (−0.44 s), the MLP blend (−1.25 s). Within one machine the
  comparisons are sound; across machines they are not distinguishable.
* **Any published result must state the thread count**, or it will not reproduce. This bears on
  the reproducibility requirement for prize eligibility.
* Compare configurations **on one machine only**, and re-baseline after moving.

### ⚠️ Timing notes in §10-11 were taken on a thermally throttled machine

The laptop is a fanless MacBook Air and throttles hard under sustained load. That is the cause
of the unexplained fit-time blowups flagged twice above — `cat_l2=1.0` at 1,671 s against
~150 s for its neighbours, round 3's trials 51 and 54 at 2,816 s and 3,210 s against a ~250 s
median, and that study's 14 h wall clock against 7.1 h of summed trial time. On the Oracle
instance every MLP epoch took 61-63 s with no drift.

**Scores are unaffected** — RMSE does not depend on how slowly a fit ran — but **every timing
figure in §10 and §11 is wrong**, including the "~4.5 min per trial" cost model and the advice
to budget 2x. Re-measure on the instance before costing any future sweep.

### The stand-occupancy "bound" — a circular constraint, recorded so it is not retried

It looks like free information: an aircraft cannot push back before the previous occupant has
vacated the stand, and arrival rows are labelled on both sides, so

    taxi = MVT_TIME − BLOCK_TIME  <  MVT_TIME − in_block(previous arrival at this stand)

is apparently a hard cap computable at prediction time. It is not.

`merge_asof` finds the most recent arrival before **takeoff**, but that arrival may have landed
*after* our aircraft already pushed back. A leaves stand 12 at 10:50, B arrives at stand 12 at
11:30, A takes off at 12:00 — the "bound" then claims A's taxi was under 30 min when it was 70.
Computing it correctly needs the previous arrival before **off-block**, which is the quantity
being predicted. **The constraint is circular.**

The data says so before any model does: **the truth violates the bound on 9.6% of validation
rows**. Applying it anyway: 276.50 → 537.69 full. Even restricted to the 113 regime rows it
costs 182 s.

Lesson: when a proposed hard constraint is violated by the ground truth at a non-trivial rate,
that *is* the result — stop there rather than scoring it.

### OpenStreetMap taxi geometry — built, measured, marginal

The only surviving external-data lead after ADS-B died. OSM airport geometry (Overpass,
**ODbL 1.0**) gives a real stand -> runway-threshold distance, and unlike ADS-B it covers all
ten airports. The joins are excellent (`e35_osm_match.py`), weighted by departures:

| | |
|---|---|
| stand `ref` -> `STAND_mvt` | **88.1%** overall; **LIRF 100%**, LTFM 99.3%, LEMD 98.9%, EGLL 98.5% |
| runway threshold -> `RUNWAY_mvt` | **100%** at every airport |
| within-airport residual correlation | **+0.210** |
| train/serve coverage | 87.4% / 88.2% — symmetric |

The runway join picks the correct *departure threshold* by bearing: for designator `D` it takes
the endpoint whose bearing to the far end is ~`D*10` degrees. Taking the wrong end would invert
the distance for every flight on that runway.

**Worth −0.44 s full / −0.48 s trimmed** (`e36_osm_model.py`), and **LIRF barely moved**
(396.2 -> 396.0) despite matching 100% there. Same magnitude as the off-block gaps, i.e. noise
against a ±6 s protocol.

Why so small, in hindsight: §8 already recorded that `stand_area x RUNWAY_mvt` as a categorical
changed nothing, because LightGBM recovers the interaction wherever data is plentiful. A numeric
distance only adds generalisation to *rare* stand-runway pairs, and those carry little weight.
**The geometry was already in the model, encoded categorically.** Kept in the build (harmless,
open-licensed, documented) but not a result.

Attribution if shipped: *"Airport geometry (c) OpenStreetMap contributors, licensed under
ODbL 1.0."*

### The ADS-B off-block project — sized, piloted, and ABANDONED in one day

`e31`/`e32`/`e33` sized a self-computed off-block as worth **60-116 s**, the only lever big
enough for the leader's 224. The pilot (`e34_detector_v1.py`, one day of adsb.lol surface data)
killed it:

| | |
|---|---|
| departures with any surface track | 25.2% |
| tracks beginning stationary (needed to see a pushback) | 18.3% of those |
| median first ADS-B point vs true off-block | **+127 s (after it left the stand)** |
| median ground speed at first sighting | **8.8 kt — already taxiing** |
| detector RMSE vs truth | **2,152.9 s** |
| `AOBT_3_flt`, identical rows | **697.9 s** |

**Aircraft are not transmitting while parked.** EDDM, with the best receiver coverage in the
sample (165,677 ground points), still sees only **39.6%** of departures before off-block — a
receiver problem would put it near 100%. So this is transponder behaviour and **no ADS-B source
fixes it, OpenSky included**: `AOBT_3_flt` is ADS-B-derived, and its +2,758 s bias on holds is a
data limit rather than an algorithmic one.

Two lessons worth keeping:

1. **A simulation can size a prize; only a pilot can tell you the prize exists.** `e32`/`e33`
   were correct that a detector of accuracy `sigma` would be worth 60-116 s. They could not test
   whether the event is observable — and it mostly is not.
2. **Coverage geography was the wrong worry.** adsb.lol also has zero surface coverage at LIRF,
   LTFM, LFPG and LEMD (68.7% of the weighted MSE), which alone capped the ceiling at 235. But
   even at perfectly covered airports the physics fails, which is the deeper reason.

See `docs/adsb_offblock_plan.md` for the full record.

### Closed off this round — do not re-litigate

- **Prediction recalibration.** Conditioned on *predicted* value the model does under-predict
  at the top (+34, +66, +115, +165 s in the top four bins), but an isotonic map fitted on one
  validation cell and applied to the other is worth **0.01 s**. The biased bins are far too
  small for their bias to matter against their variance. The whole calibration family is dead.
- **Signed off-block gap features** (`e23_offblock_gaps.py`). All ten pairwise gaps among
  {SCHED, IOBT, LOBT, EOBT_1, AOBT_3} plus seven arrival-side analogues — 17 features aimed
  directly at the hold, including `LOBT − IOBT`, the closest thing in the file to an ATFM slot
  delay. Worth **−0.52 s full / −0.61 s trimmed**. The mechanism is real but the model was
  already extracting it. Harmless to keep, not a result.
- **Gating the regime branch on a placeholder classifier** (`e25`, `e26`). This one looked
  like the round's big win and is the most useful negative here. **47.9% of the 1,507
  missing-aircraft-type rows carry a placeholder `BLOCK_TIME_UTC_mvt` exactly equal to
  `SCHED_TIME_UTC_mvt`**, and for those `proxy_taxi_sched` predicts the target with an RMSE of
  **4 seconds** — and they are the *long* rows (median target 7,534 s against 1,030 s for the
  rest). A classifier detects them out of sample at **AUC 0.75–0.94**. It still buys nothing:

  | predictor (leave-one-month-out, count-weighted) | RMSE |
  |---|---|
  | **ORACLE gate** (mode known exactly) | **3,672 s** |
  | delta model (shipped) | 4,240 s |
  | hard gate, p > 0.9 | 4,248 s |
  | soft blend `p·proxy + (1−p)·delta` | 4,341 s |
  | hard gate, p > 0.5 | 4,371 s |

  The shipped delta model already predicts the placeholder rows at 3,063 s, so perfect mode
  knowledge is worth only 13% of the regime's MSE — **9 s** of total weighted RMSE — and no
  realizable classifier captures any of it.

### What worked — `experiments/e20_config.py` + `e27_config_model.py`

The one family round 2 identified and never built. `e18_gaps.py` was written on 2026-09-13,
tested three candidate families, and its findings were never recorded anywhere — this round
built two of them.

| arm | full | trimmed |
|---|---|---|
| baseline (127 features, r3 parameters) | 279.82 | 235.62 |
| **+ runway configuration & stand rotation (12)** | **278.46** | **233.75** |
| **+ 3-seed bag** | **277.62** | **232.75** |

**Trimmed moves more than full in both arms**, which is the check that separates these from
round 2's `e9_verify_boost.py` trap: the gain is on the bulk of normal rows, not on one lucky
tail row. A tail artefact shows the opposite signature — full moving while trimmed goes
backwards.

**Runway configuration** (`depcfg_*`, `arrcfg_*`) — which set of runways the airport was
actually working in a ±30 min window around this takeoff, as an airport-namespaced categorical,
plus how many runways were in use, this flight's runway's share of the movements, the total
movement count, and whether the configuration differed 60 minutes earlier. Departure and
arrival configurations are kept separate: the departure set fixes this flight's route, the
arrival set fixes what it has to cross or give way to. This is a first-order driver of taxi
distance — two departures from the same stand to the same runway travel different routes
depending on which other runways are active — and nothing in the previous 127 expressed it.

**Stand rotation** (`stand_turnaround_sec`, `stand_prev_arr_taxi`) — how long the inbound that
last vacated this stand had been parked on it, via `merge_asof` on (airport, stand) against
arrival in-block times. The ARR rows are labelled on both sides, so in-block is
`MVT_TIME + TAXITIME` and the feature is computable identically at prediction time.

Train/serve symmetry is clean: medians match within 1% on every column, and only **0.06%** of
ranking rows sit on a departure configuration never seen in training.

**Seed bagging** is the free lunch — averaging fits that differ only in `random_state` cuts
variance without touching bias. Worth −0.84 s full / −1.00 s trimmed for three seeds, at three
times the fit cost. It also reduces *measurement* noise, which matters when the effects being
chased are smaller than the protocol's ±6 s.

- **Piecewise-linear leaves and monotone constraints** (`e28_linear_tree.py`). The last
  in-model idea, aimed at the saturation E22 found: a constant leaf cannot express "keep
  following the proxy", so `linear_tree=True` should let the prediction track a numeric feature
  past the values the leaf was fitted on. Measured at a matched 1,200-iteration budget:

  | arm | full | trimmed | tail (y ≥ 1,800 s, n=15,041) |
  |---|---|---|---|
  | constant leaves (reference) | 284.19 | 239.98 | 747.6 |
  | + monotone constraints | 284.66 | 240.51 | 747.8 |
  | linear leaves | **506.77** | **484.59** | **1,114.3** |
  | linear leaves + monotone | **565.50** | **546.17** | 1,021.2 |

  Not merely unhelpful — **catastrophic, and worst on the tail it was meant to fix**. The
  extrapolation linear leaves enable runs away rather than tracking the proxy; constraining it
  to be monotone in the proxies makes it worse still. Monotone constraints alone are flat to
  slightly negative. The saturation is real but it is not fixable inside the tree.

> ⚠️ **Correction to §8.** The "38.3 s headroom" quoted for the missing-aircraft-type regime
> comes from an oracle that knows the *target*. The reachable ceiling, from an oracle that
> knows the *mode*, is **9 s**, and the realizable figure measured this round is **0 s**. Plan
> against 9 s, not 38 s. The regime is effectively at its floor; the gap to the leaders has to
> come from the normal rows, where trimmed RMSE is 235.6 s and the leaders' implied figure is
> around 216 s.

### Loss, target scale and target encoding — `experiments/e43_residuals_loss_te.py` (2026-09-27)

Run on the Oracle instance (4 threads), one seed of the exact v5 configuration, post-processing
unchanged. Compare within this table only.

| arm | full | trimmed | Δ full | Δ trimmed |
|---|---|---|---|---|
| **ref (v5, 1 seed)** | **274.40** | **230.92** | — | — |
| + 6 OOF target encodings (month-grouped folds, m=100 smoothing) | 273.73 | 230.35 | −0.67 | −0.57 |
| Huber, δ = 600 s | 275.76 | 232.39 | +1.36 | +1.47 |
| Huber, δ = 1,200 s | 274.30 | 230.69 | −0.10 | −0.23 |
| target winsorised at train p99.5 | 285.31 | 239.71 | +10.91 | +8.79 |
| target winsorised at train p99.9 | 277.34 | 232.30 | +2.94 | +1.38 |
| log1p target, naive expm1 | 280.26 | 237.12 | +5.86 | +6.20 |
| log1p target, Duan smearing (×1.0135) | 278.99 | 235.72 | +4.59 | +4.80 |

**Residual breakdown of ref** (share of leaderboard-weighted MSE, by *true* taxi time): <10 min
6.1% (bias +104 s), 10-20 min 28.9% (+31 s), 20-30 min 19.9% (−84 s), 30-60 min 19.9% (−381 s),
60+ min 25.1% (−1,853 s). By airport LIRF alone is 42.1% (RMSE 637 s vs 160-260 s elsewhere);
per-airport bias is ~0 everywhere. By local hour the error is flat — no congestion-peak signature.
By slice July is 71.8%.

Closed — do not re-litigate:
- **Robust losses and target compression.** Every transform that shrinks the tail during fitting
  loses, and the more it shrinks the more it loses. The metric is squared error; the model is
  already fitting the conditional mean it rewards.
- **OOF target encoding** duplicates what native categoricals + `cat_smooth=100` already extract.
  Sub-noise, same class as the off-block gaps and OSM geometry.

### LIRF forensics, per-airport models, LIRF-only cap — `experiments/e44_lirf_per_airport.py` (2026-09-27)

Same machine and reference as e43 (ref 274.40 / 230.92).

| arm | full | trimmed | Δ full |
|---|---|---|---|
| ten per-airport models (v5 params) | 276.03 | 231.96 | +1.63 |
| global, LIRF rows replaced by a LIRF-only model | 275.52 | 231.81 | +1.12 |
| LIRF-only target cap at LIRF p99.5 (3,714 s, 384 rows) | 276.10 | 232.17 | +1.70 |
| LIRF-only target cap at LIRF p99.9 (5,029 s, 78 rows) | 274.81 | 231.14 | +0.41 |

Per airport, own models land within ±2 s of the global model everywhere except LIRF (+6.2 s):
the smallest airport loses most from giving up pooled strength.

**The (airport × month) matrix under v5** (`e44_xtab.csv`): `jul LIRF` is 35.5% of weighted MSE,
22.6 points of it from 337 missing-aircraft-type rows; on normal rows alone it is still the worst
cell at 466 s (Jan 311 s). July is broadly harder, not just at Rome — LEBL 175→264, EGLL 228→287,
LTFM 216→273 on normal rows.

**Not a regime change at LIRF.** Mean taxi runs 1,033 s (Feb) → 1,288 s (Jul) → 1,042 s (Nov) with
no step; runway 25 carries ~90% of departures every month; missing-type rows rise 58 → 337 → 52
on the same seasonal curve.

**The label is noisier at LIRF and LTFM.** `AOBT_3_flt − BLOCK_TIME_UTC_mvt`, train months:

| airport | median | IQR | \|gap\| > 600 s |
|---|---|---|---|
| LIRF | +117 s | 419 s | 12.6% |
| LTFM | −301 s | 421 s | 17.7% |
| other eight | −109…+32 s | 238–357 s | 1.6–6.7% |

Part of the LIRF/LTFM error is therefore label noise that no feature can remove — yet capping
LIRF's tail still loses, so the long LIRF taxis are signal, not recording artefacts.

Closed — do not re-litigate: per-airport models, airport-scoped target caps, a "LIRF & July"
indicator (the model already has airport × day-of-year; there is no discontinuity to flag).

### METAR convective features — `experiments/e45_metar_convective.py` (2026-09-27)

Source: IEM's public-domain METAR archive (vetted in §1). 17 features from reports at or before
takeoff (`src/metar_features.py`): TS / VCTS / CB / TCU / heavy precip / hail now, TEMPO TS in
the trend, convective-in-2h, counts over 3 h / 6 h, minutes since last convective report,
reports per hour (SPECI proxy), visibility, minimum RVR, gust. Train/serve rates match
(convective-now 1.67% train vs 1.31% ranking; TS-now 0.13% vs 0.10%).

**Sizing first (no fit, v5 residuals).** Rows with a convective report in the previous 2 h are
3.5% of rows and 8.7% of the weighted MSE — RMSE 432 s vs 267 s, bias −40 s (−60 s with a TS
in the past 6 h). If every convective row matched its own (airport, slice) cell's
non-convective error, full RMSE would fall 274.40 → 266.61: **7.8 s is the ceiling**. It does
not explain `jul LIRF` — that cell's non-convective rows alone are 742 s and 32.4% of the MSE.

**Measured (Oracle VM, 1 seed, vs e43 ref):** **273.89 full / 230.35 trimmed, −0.51 / −0.57 s.**
The mechanism is real and lands where aimed — convective-row RMSE 432.2 → 424.5 with bias
−39.7 → −17.3 s, TS-6h rows 516.2 → 505.1, non-convective rows unchanged (266.9 → 266.8) —
but the rows are too rare for it to matter. Best-ranked feature is 95th of 177 by gain; the
family totals ~0.3% of gain. Same class as the off-block gaps, OSM geometry and OOF target
encoding: genuine, sub-noise, not a result on its own.

### Missing-type autopsy and noise-aware sample weights — `experiments/e46_noact_autopsy_weights.py` (2026-09-27)

**The 428 validation missing-aircraft-type rows (337 `jul LIRF`) are failed NM joins:**
`dq_no_flight_record` on 100%, `AOBT_3_flt` absent on 100%; stand 99.5% and runway 100% present.
Target p50 1,646 s, p75 7,640 s, p90 12,786 s, p99 81,032 s. Exact `BLOCK == SCHED`
placeholders are 4.9% of these validation rows (vs 47.9% of the training regime, `e24`).

Substituting a baseline on those rows only:

| predictor | RMSE (428) | RMSE (jul LIRF 337) | total full |
|---|---|---|---|
| **shipped regime branch** | **3,885** | **4,089** | **274.40** |
| `proxy_taxi_sched` | 5,550 | 6,089 | 309.22 |
| missing-type train mean | 11,895 | 12,007 | 484.38 |
| main model, no branch | 12,767 | 12,753 | 511.17 |
| airport-hour median of normal rows | 13,017 | 12,986 | 518.99 |
| target oracle | 0 | 0 | 236.76 |

A "dumb fallback" is catastrophic here, not a safe bank: the labels are hours long, and a
normal-taxi median misses each one by hours. The branch is already 3x better than any baseline;
the reachable headroom remains the 9 s mode-oracle figure from `e26`, not the 37.6 s target oracle.

**Noise-aware sample weights** (0.25 weight; gap = `AOBT_3 − BLOCK` centred on each airport's
training median, since LTFM's −301 s median is an offset, not noise):

| arm | down-weighted | full | trimmed | Δ full |
|---|---|---|---|---|
| \|centred gap\| > 600 s at LIRF + LTFM | 20,710 | 278.55 | 235.71 | **+4.15** |
| centred gap < −600 s anywhere (ADS-B moves before off-block) | 14,227 | 274.68 | 231.28 | +0.28 |

The symmetric version loses 4 s because large *positive* gaps are the real post-pushback hold
(`e22`), which is scored — down-weighting them is winsorising by another name. The
physically-contradictory negative side is flat: whatever noise it marks, the model was already
averaging over it. Closed.

### The missing-type branch itself — `experiments/e47_regime_branch_variants.py` (2026-09-27)

**Correction to the placeholder figures.** Exact `BLOCK == SCHED` is 2–12% of missing-type
rows in *every* month (≈6.5% Jan–Jun, 4.9% of the validation rows); within ±60 s it is 32–76%
by month, and the median `BLOCK − SCHED` on these rows is 5 s. The 47.9% quoted from `e24`
refers to a looser definition. There is no train/serve placeholder skew; there is a seasonal
one — July has the fewest near-placeholders (32%) and the most genuine extremes.

**Profile.** 1,487 of 1,507 are LIRF; 99.9% IFR; callsigns are scheduled carriers (RYR 221,
WMT 157, ITY 83, WZZ 46, EJU 35 …); stand-area and runway mix close to normal LIRF traffic.
Not a GA / VFR / charter segment — ordinary airline flights whose NM join failed.

Scored by leave-one-month-out over all 12 months on the branch's own rows (count-weighted),
plus the mirrored total with e43's main-model predictions elsewhere:

| branch variant | LOMO RMSE | LOMO Jul | mirrored full | months beating base |
|---|---|---|---|---|
| **base (shipped)** | **4,230** | **3,997** | **274.40** | — |
| trained without exact placeholders | 4,251 | 3,993 | 277.89 | 4/12 |
| trained without ±60 s placeholders | 6,134 | 4,883 | 298.50 | 0/12 |
| Huber δ = 600 s on the delta | 4,449 | 4,493 | 280.79 | 4/12 |
| Huber δ = 1,800 s on the delta | 4,314 | 4,233 | 274.68 | 6/12 |
| log1p(y) + Duan smearing | 6,499 | 6,807 | 395.97 | 0/12 |
| + peer disruption (`MVT − SCHED` of other departures ±60 min) | 4,345 | 4,370 | 279.33 | 4/12 |

Every variant loses. Placeholders are not contamination: roughly half the rows *scored* are
near-placeholders too, and the zero-delta points are what teach the branch to follow
`proxy_taxi_sched` on them — remove them and it fails on exactly those rows (6,134 s). Log and
Huber lose here for the same reason as globally: the metric is squared error and the extremes
are scored. Peer disruption is uninformative because it is ubiquitous — 94.2% of missing-type
rows have at least one peer over an hour late — so it does not separate the systemic cases.

The branch is at its floor for the information available. Closed.
