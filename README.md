# Taxi-out time prediction — PRC Data Challenge 2026

Entry for the [PRC Data Challenge 2026](https://ansperformance.eu/study/data-challenge/dc2026/),
run by EUROCONTROL's Performance Review Commission in partnership with the OpenSky Network.
Team **`diligent-fireplace`**.

**Task.** Predict `TAXITIME_SEC_mvt` — the time from off-block to takeoff, in seconds — for
departing flights at ten major European airports, scored by RMSE. Training data is all of 2025.
The leaderboard phase scored January and July 2026; the **final phase** scores a single submission
over **January, February, June and July 2026** (670,790 departures), ranked on Jan/Jul, Feb/Jun and
all four months combined.

**Result.** Leaderboard (Jan/Jul 2026) RMSE **279.73 s** (v5), against a 469.9 s per-airport-median
baseline. Progression: v2 329.76 → v3 287.10 → v4 282.23 → v5 279.73.
The **final submission** (`diligent-fireplace_final.parquet`) is the v5 configuration, unchanged,
with every ranking-side feature rebuilt from the four-month `final_ranking.parquet`; nothing in it
was tuned to February or June. Its score is published by the organizers.

---

## Quickstart

```bash
brew install libomp                     # macOS only; LightGBM needs OpenMP
python -m venv .venv && source .venv/bin/activate   # Python 3.14
pip install -r requirements.txt

# place the challenge parquet files in Dataset/ (see "Data" below), then:
python scripts/fetch_external.py            # weather + holidays -> data/external/
python scripts/build_features.py            # feature tables     -> data/processed/
python scripts/build_surface_features.py    # rolling surface state
python scripts/build_osm_features.py        # taxi geometry from the committed OSM extracts
python scripts/build_config_features.py     # runway configuration + stand rotation
python scripts/build_submission.py --team <team> --version <N>
```

That reproduces the final submission: `build_submission.py` fits on all twelve 2025 months with
the committed hyperparameters (`data/processed/lgbm_best_params_r3.json`, 3,546 rounds, 3 seeds,
160 features) and writes `submissions/<team>_v<N>.parquet`. The final phase requires the name
`<team>_final.parquet`, so copy it to that name before upload (`scripts/upload_submission.sh`).

Which scoring file is used is set in one place, `src/cleaning.py` `ranking_path()`: it defaults to
`final_ranking.parquet` / `final_submitting.parquet`, and `PRC_RANKING=ranking` switches every
builder back to the Jan/Jul leaderboard files. Rebuild **all** ranking-side feature tables after
switching — the congestion, surface-state and configuration features count surrounding traffic in
the scoring file itself.

`python scripts/train_baseline.py` / `scripts/train_lgbm.py` run the out-of-time validation
protocol described below. Timing: the three-seed all-data fit takes ~45 min on a 4-core ARM
cloud VM; the feature builds take a few minutes. LightGBM results move by ~1.5 s with thread
count (`SKILLS.md` §11), so expect agreement to that precision rather than bit-for-bit.

## The problem, and the one fact that shapes everything

`TAXITIME_SEC_mvt` is exactly `MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt` (a 100.000 % match on
training departures). In the scoring file `BLOCK_TIME_UTC_mvt` is **100 % blank** while
`MVT_TIME_UTC_mvt` — the actual takeoff — is **100 % present**.

So the task is not "how long will this aircraft taxi?" but **"given that it took off at this
instant, when did it leave the stand?"** Two consequences run through the whole design:

1. Anything derived from the off-block time is unusable, but the **takeoff time is a legitimate
   anchor**. That is what makes the congestion and surface-state features computable at
   prediction time at all.
2. The best single predictor is `AOBT_3_flt`, an independent ADS-B-derived off-block time
   present on 98.5 % of scored rows. The model is largely learning the *disagreement* between
   that and the airport's own block record — a quantity with σ ≈ 384 s and strongly
   airport-specific bias.

## Method

A single **LightGBM** regressor on **160 features** (3-seed bag, `cat_smooth=100`), plus two
post-processing rules for row populations the regressor provably cannot learn. Full specification
in [`SKILLS.md` §6](SKILLS.md), with the round-4/5 additions in §11; the 127 core features are
catalogued with their measured importance in [`docs/features.md`](docs/features.md). The other 33
are runway-configuration and stand-rotation features (`src/config_features.py`, 12), off-block gap
features (`src/offblock_gaps.py`, 17) and OpenStreetMap taxi geometry (`src/osm_features.py`, 4).

Feature families of the 127-feature core, by share of model gain:

| family | gain % | what it measures |
|---|---|---|
| off-block proxies | 44.5 | takeoff minus each available off-block estimate |
| stand / apron | 18.5 | where the aircraft started — the taxi-distance proxy |
| raw movement & flight-plan fields | 11.6 | runway, airline, aircraft type, airport |
| surface state | 10.6 | how long taxis are *actually* taking right now |
| off-block corroboration | 5.4 | how far the independent off-block estimates agree |
| route, weather, congestion, time, other | 9.4 | |

Three things are worth singling out, because they are where the gains came from rather than
from model tuning:

- **Surface state.** Rolling means of *realised* durations from movements already completed —
  including recent **arrival** taxi-in times, which are fully labelled in the scoring file and
  were otherwise unused. Counts of surrounding traffic never helped; realised durations did.
  Every window is backward-looking and half-open, so a flight can never see its own outcome.
- **A separate model for unmatched records.** Rows with a null `AIRCRAFT_TYPE_mvt` carry no
  flight-plan record at all, so the dominant feature is missing on 100 % of them. They are a
  different population (mean 6,451 s against ~990 s) and get their own small LightGBM,
  regressing `proxy_taxi_sched − taxi` rather than the target directly.
- **A corroborated-proxy override.** A gradient-boosted tree cannot predict outside the range
  of targets it was fitted on, so a genuinely 24-hour recorded taxi is predicted at ~1,000 s.
  Usually that is right. But when the scheduled, initial, last *and* estimated off-block times
  all sit at the same implausible offset, the record's date is shifted as a whole and the proxy
  is correct. The rule separates the 19 training cases perfectly and fires on 2 scored rows.

### Validation

The contest is an **out-of-time forecast**, so a random holdout would be meaningless. Training
is on January–June 2025; validation is on December 2025 (standing in for January) plus July
2025, with every (month, airport) cell re-weighted onto the share the real scoring set gives
it. Anomalous rows are never dropped from the validation set — they are a large part of what is
actually scored. Details and the known compromises are in [`SKILLS.md` §5](SKILLS.md).

The protocol has tracked the leaderboard to within ~6 s across four submissions (v2 predicted
332 → scored 329.76; v3 281.5 → 287.10; v4 276.50 → 282.23; v5 275.27 → 279.73).

**Generalisation to February and June.** The final phase adds two months the leaderboard never
scored. The shipped model has no month-specific component: it is fitted on all twelve 2025
months (so February and June 2025 are in its training data), its post-processing rules key on
row properties rather than dates, and its hyperparameters were chosen once, on the Jan/Jul
protocol, then frozen. On the 344,841 departures common to both scoring files, the four-month
rebuild differs from v5 by 17.6 s RMS with a mean shift of −0.04 s — rebuilding moved
individual predictions slightly but did not bias them. No February/June hold-out score has been measured.

## Repository layout

```
src/cleaning.py          loading, cleaning rules, DQ flags, off-block proxies + corroboration;
                         ranking_path() selects the scoring file
src/features.py          congestion, queue, sequence, time, route, stand, wind, weather regimes
src/surface_state.py     backward-rolling realised-taxi state
src/config_features.py   runway configuration in use + stand rotation (turnaround) features
src/offblock_gaps.py     gaps between the independent off-block estimates
src/osm_features.py      stand-to-runway taxi geometry from OpenStreetMap
src/metar_features.py    METAR weather features (experiment e45 only; not in the shipped model)
src/external_data.py     Open-Meteo fetch/cache/join, holiday calendar
src/airports.py          airport reference table (coordinates, timezone, holiday subdivision)
src/validation.py        composition-mirroring split and weighted RMSE
src/postprocess.py       the two post-processing rules
scripts/                 one entry point per pipeline stage (see Quickstart)
scripts/fetch_*.py       external-data fetchers; fetch_adsb_ground.py / probe_opensky.py belong to
                         the abandoned ADS-B off-block pilot (SKILLS.md §11)
experiments/             one script per hypothesis tested, including the negative results
data/external/           provenance sidecars (*.meta.json) and the OSM extracts (osm/*.json)
data/processed/          built caches (gitignored) + the committed tuned hyperparameters and sweep log
docs/data_dictionary.md  all 31 delivered columns and their coded values
docs/features.md         the 127 core model inputs with measured importance
docs/adsb_offblock_plan.md  the ADS-B off-block detector: plan, pilot, and why it was abandoned
analysis.ipynb           exploratory analysis and the cleaning / external-data walkthrough
docs/findings.md         the measured findings behind the design ("finding N" in code comments)
SKILLS.md                design decisions and the measurement behind each one (the lab notebook
                         the code's "§N" references point into)
```

`experiments/README.md` maps each script to the question it settled. The negative results are
kept deliberately — they are the cheapest part of the work to reuse.

## Data

**Challenge data is not redistributed here.** `Dataset/` is gitignored; obtain the monthly
`training_*.parquet`, `final_ranking.parquet` and `final_submitting.parquet` files (and, for the
leaderboard phase, `ranking.parquet` / `submitting.parquet`) from the organizers under their own
terms and place them there.

All supplementary data is openly licensed, as the challenge rules require:

| source | used for | licence |
|---|---|---|
| [Open-Meteo](https://open-meteo.com) historical forecast API | hourly weather per airport | CC BY 4.0 |
| [`holidays`](https://github.com/vacanza/holidays) | public-holiday calendars, by country and subdivision | MIT |
| [OurAirports](https://ourairports.com/data/) | airport coordinates and elevation | Public domain (Unlicense) |
| [OpenStreetMap](https://www.openstreetmap.org) via the Overpass API | runway, taxiway and stand geometry → taxi-distance features | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) |
| [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml) ASOS/METAR archive | METAR weather — tested in `experiments/e45`, **not used by the shipped model** | Public domain |

Fetched artefacts under `data/external/` each carry a `.meta.json` provenance sidecar recording
source, endpoint, licence, retrieval timestamp, and the exact variables and date range pulled.
Regenerate them with `python scripts/fetch_external.py` (weather, holidays) and
`python scripts/fetch_metar.py`. The OpenStreetMap extracts are **committed** as
`data/external/osm/*.json`, because the Overpass API returns today's map rather than the one the
shipped features were built from; `python scripts/fetch_osm_airports.py` refetches them. Those
files are a database extract licensed under the ODbL, separately from this repository's GPLv3 code.

> **Weather data by [Open-Meteo.com](https://open-meteo.com), licensed
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).**
>
> **Airport geometry © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
> available under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).**
>
> **METAR data courtesy of the Iowa Environmental Mesonet, Iowa State University.**

The ADS-B ground data used by the abandoned off-block pilot (`scripts/fetch_adsb_ground.py`,
OpenSky Network) is **not** used by the model and is not redistributed here.

Sources considered and **rejected** on licensing grounds are logged in
[`SKILLS.md` §1](SKILLS.md) — notably EUROCONTROL's own PRU taxi-out dataset, which is
topically perfect but carries non-commercial and no-modification terms, and Meteostat (CC BY-NC).

## Licence

Released under the **GNU General Public License v3.0** — see [`LICENSE`](LICENSE). Publication
under GPLv3 is a condition of prize eligibility for this challenge.
