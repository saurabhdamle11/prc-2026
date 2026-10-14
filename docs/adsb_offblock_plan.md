# Deriving our own off-block time from raw ADS-B — scoping note

**STATUS: ABANDONED 2026-09-26 after the pilot. The premise is physically false.**

The pilot ran, and it killed the project in about an hour. Kept in full because the reasoning
was sound and the negative result is worth more than the plan was.

**ADS-B cannot observe the off-block moment for most departures, from any source.** Measured on
2025-01-15 across the six airports adsb.lol covers on the surface (`experiments/e34_detector_v1.py`):

| | |
|---|---|
| departures with any surface track | **25.2%** |
| of those, tracks beginning *stationary* (the precondition for seeing a pushback) | **18.3%** |
| median first ADS-B point relative to true off-block | **+127 s** — *after* it has already left |
| median ground speed at first sighting | **8.8 kt — already taxiing**; only 15.9% stationary |
| **detector RMSE vs true BLOCK_TIME** | **2,152.9 s** |
| **`AOBT_3_flt` RMSE, identical rows** | **697.9 s** |

The detector is **3x worse than the feature we already have**, and worse in every taxi band.

**This is transponder behaviour, not receiver coverage, which is why no better source fixes it.**
EDDM has 165,677 ground points — the best coverage in the sample — and still sees only **39.6%**
of departures before off-block. A receiver problem would show EDDM near 100%. Aircraft are not
transmitting while parked; the transponder is selected on at or after pushback.

**Consequence for OpenSky.** Do not pursue Trino access for this purpose. `AOBT_3_flt` is itself
ADS-B-derived, and its +2,758 s bias on held flights is a *data* limit, not an algorithmic one —
which also explains why it is nearly unbiased on ordinary taxis (short push, transponder already
on) and catastrophic on long holds (extended silent wait). Better receivers cannot make a silent
transponder audible.

**What was wrong with the sizing simulations.** `e32`/`e33` modelled a detector that *observes*
off-block with error `sigma`. They were internally sound and correctly said that such a detector
would be worth 60-116 s. The assumption they could not test was whether the event is observable
at all. It mostly is not. **A simulation can size a prize; only a pilot can tell you the prize
exists.**

---

*Original plan retained below.*

**Promoted from side bet to main line on 2026-09-26**, when the leader was observed at 224 s:
v4's trimmed RMSE (every anomalous row solved perfectly) is 237.2 on the board, so the tail
cannot close that gap and no amount of regime work will. See `SKILLS.md` §11.
This is the "week 2" item from the round-4 plan. It is written down now so the decision to
start or abandon it is made against a spec rather than a hunch.

## Why this and not more features

The whole task reduces to *infer `BLOCK_TIME_UTC_mvt` given `MVT_TIME_UTC_mvt`* (`SKILLS.md`
critical finding 1). `AOBT_3_flt` is the best available estimate of that, supplies 17.8% of
the model's split gain, and `experiments/e22_calibration.py` proved it **inverts in the tail**:

| true taxi | actual | `MVT − AOBT_3` | `MVT − EOBT_1` | `AOBT_3 − BLOCK` |
|---|---|---|---|---|
| [0, 1200) | 833 | 873 | 1,144 | −67 |
| [1800, 2700) | 2,008 | 1,435 | 2,401 | +478 |
| [3600, ∞) | 4,312 | **1,079** | 4,014 | **+2,758** |

`AOBT_3_flt` marks the first detected *movement*; the target is measured from *off-block*. An
aircraft that pushes back and then holds on the apron has its entire hold skipped. Normal rows
with a true taxi ≥ 1,800 s are **4.2% of rows and 30.1% of the weighted MSE**, so this single
definitional mismatch sits on top of the largest remaining error population.

Round 4 established that this is not recoverable from the delivered columns: signed pairwise
gaps among all five off-block estimates were worth −0.52 s, and prediction recalibration 0.01 s
(`SKILLS.md` §11). The information is not in the file. It is in the raw trajectory.

## How accurate does the detector have to be? — `experiments/e31_offblock_ceiling.py`

Because `TAXITIME = MVT_TIME - BLOCK_TIME` is an exact identity, an off-block estimator with
RMS error `s` yields a taxi predictor with RMS error `s`. Training rows carry `BLOCK_TIME`, so
every existing estimate can be scored against the truth on the protocol's own weighting:

| off-block estimate | coverage | weighted RMSE vs truth | median bias | p90 abs err |
|---|---|---|---|---|
| **`AOBT_3_flt`** (ADS-B, what the model leans on) | 98.6% | **422.9 s** | −28 s | 536 s |
| `EOBT_1_flt` | 98.6% | 753.3 s | −306 s | 1,196 s |
| `LOBT_flt` / `IOBT_flt` | 98.6% | 825 s | −324 s | 1,315 s |
| `SCHED_TIME_UTC_mvt` | 100% | 2,587.7 s | −537 s | 2,861 s |
| *the v4 model, combining all of them* | 100% | **231.5 s** | — | — |

Two readings, both important. `AOBT_3_flt` is **nearly unbiased but heavy-tailed** — exactly the
signature of missing apron holds. And the model already **halves** its error by correcting
systematic per-airport, per-stand and per-operator bias, which means it would exploit a better
estimate too.

**Target: an RMS error of roughly 150-200 s**, against `AOBT_3_flt`'s 422.9 s. At 150 s the
board score lands near 218 even with the anomalous rows handled no better than today. That is a
2-3x improvement on the supplied proxy — plausible for a purpose-built detector, since ADS-B
surface tracks resolve movement onset to tens of seconds wherever receiver coverage is good.

> **Coverage is the real risk, not precision.** A detector that is excellent on 70% of flights
> and absent on the rest blends back toward `AOBT_3_flt` and gains far less than the headline
> suggests. Measure ground-coverage fraction per airport in the very first pilot pull, before
> tuning any detection logic — a low number at LIRF in particular would gut the whole thesis,
> since `jul LIRF` alone is 35.6% of the metric.

## What would be built

A pushback detector over OpenSky state vectors: for each departure, find the last timestamp at
which the aircraft was stationary at its stand before the taxi began, rather than the first at
which it exceeded a movement threshold. The difference between those two definitions is
exactly the hold that `AOBT_3_flt` discards.

Output: one extra column per DEP row, `offblock_adsb`, joined like any other feature, plus
`proxy_taxi_adsb = MVT_TIME − offblock_adsb`. It does not replace `AOBT_3_flt`; the model gets
both and decides.

## The join key — CHECKED 2026-09-26, and it works

The one thing that could have killed this outright. The challenge data is anonymised, but only
``AIRCRAFT_OPERATOR_flt`` is hashed. **``CALLSIGN_flt`` carries real ICAO callsigns** —
``DLH8UL``, ``THY9GK``, ``RJA126`` — on **98.5%** of ranking DEP rows and **99.1%** of training
DEP rows. OpenSky's ``callsign`` field uses exactly that format, so the join is
``(callsign, airport, time window)`` and needs no de-anonymisation.

## ⚠️ SOURCE CHANGED: OpenSky Trino is NOT usable — adsb.lol is

**OpenSky's historical Trino database cannot be used for this contest.** It is free for
approved researchers, but it is **not open-licensed**, and the challenge requires every
supplementary dataset to be *openly accessible, open-licensed and documented* as a hard
prize-eligibility condition (challenge eligibility rules). Free-for-researchers is not the
same as open, and the distinction decides eligibility. Everything below about Trino limits is
retained only as background.

**Use adsb.lol instead.** It publishes its complete historical archive under **ODbL 1.0**
(with CC0-1.0 also offered) as public GitHub releases — no access request, no eligibility
review, no gatekeeper:

* <https://github.com/adsblol/globe_history_2025> and `_2026`
* <https://www.adsb.lol/docs/open-data/historical/>

Attribution required in any published output:
*"Historical ADS-B data by adsb.lol, licensed under ODbL 1.0."*

**It is also a better technical fit.** Files are per-aircraft-day gzipped JSON in wiedehopf's
readsb trace format, whose entries are::

    [seconds_after_timestamp, lat, lon, altitude, gs, track, flags, vert_rate, aircraft, ...]

`altitude` is the **literal string `"ground"`** on the surface and `gs` is ground speed in
**knots** — the exact pair a pushback detector needs, at trace resolution, with no SQL, no
scan limits and no concurrency caps.

**Verified available for all four target months** (2026-09-26):

| month | GB/day | 31 days |
|---|---|---|
| Jan 2025 | 2.23 | 69 GB |
| Jul 2025 | 3.55 | 110 GB |
| Jan 2026 | 3.17 | 98 GB |
| Jul 2026 | 4.11 | 127 GB |

~404 GB for all four months; ~315 GB if the 2025 calibration window is trimmed to two weeks
each. **Against ~72 GB of free disk**, so `scripts/fetch_adsb_ground.py` streams each day
through `tarfile` without unpacking, keeps only ground points inside the ten airport boxes, and
deletes the archive — peak disk stays at about one day.

**The remaining risk is unchanged and is still coverage, not precision.** adsb.lol is
community-fed, so its surface coverage is likely below OpenSky's aggregated network,
particularly outside western Europe (LTFM). The pilot measures this before anything else.

## Trino: the real constraints — background only, source no longer used

<https://openskynetwork.github.io/opensky-api/trino.html>

**Connection** — server `https://trino.opensky-network.org`, catalog **`minio`**, schema
**`osky`**, browser-based external authentication. Usernames are stored lowercase; use
lowercase everywhere. Python clients: `pyopensky` (<https://mode-s.org/pyopensky/>) or
`traffic` (<https://traffic-viz.github.io/>), both of which speak Trino natively.

**`state_vectors_data4`** — the table we need. Relevant columns: `time` (Unix seconds),
`icao24`, `callsign`, `lat`, `lon`, `velocity` (**m/s**), `heading`, `vertrate`, `onground`,
`baroaltitude`, `geoaltitude`, `lastcontact`, and **`hour`** (Unix hour, the partition column).

**Enforced limits, and they shape everything:**

| limit | value |
|---|---|
| scan per query | **100 GB** |
| query duration | 30 minutes |
| concurrent queries | 2 (plus 2 queued) |
| partition filter | **mandatory** — "Always filter on the partition column. Always." |

The docs also say explicitly: *"No large global bulk downloads via ad-hoc queries."* A naive
"give me fourteen months of European ground tracks" plan violates both the scan limit and that
instruction.

**The cost driver is hour-partitions, not airports.** `lat`/`lon` are not partition columns, so
a query restricted to one airport's bounding box still scans the whole global partition. Cost
is therefore proportional to *hours queried*, and the ten airports come almost free once an
hour is being read.

| scope | hour partitions |
|---|---|
| Jan-2026 + Jul-2026 (the scored months — mandatory) | 1,488 |
| full 2025 (all training labels) | 8,760 |
| **everything** | **10,248** |

**Scoping decision: do not pull all of 2025.** Take **Jan + Jul 2025** as the matched-season
training window (1,464 h) alongside the two scored months — **2,928 partitions rather than
10,248**, a 3.5x reduction for training data in the seasons that are actually scored. This is
affordable because the detector largely *replaces* the model for rows it covers: with
`taxi = MVT_TIME − offblock_adsb` computed directly, the labelled data is needed to calibrate
residual bias, not to fit a model from scratch.

**Known data gaps — measured against our own scored rows.** July 2026 and all of 2025 are
clean. January 2026 has two outages:

| outage (UTC) | scored DEP rows lost |
|---|---|
| 2026-01-02 19:00 → 2026-01-03 18:00 | 4,894 |
| 2026-01-24 08:00 → 2026-01-24 17:00 | 2,763 |
| **total** | **7,657 = 2.22% of 344,841** |

Spread evenly across all ten airports (LIRF only 575). Those rows fall back to `AOBT_3_flt`,
which is what happens today anyway — so the gaps cost essentially nothing.

> ⚠️ **Access eligibility is a genuine risk.** The application at *My OpenSky → Request Data
> Access* is reviewed, and the docs say "universities, governments, and aviation authorities
> qualify". An unaffiliated participant may not clear that bar on the form alone — the request
> must cite **PRC Data Challenge 2026** participation explicitly, and it is worth asking on the
> challenge Discord for the route the organizers intend, rather than only submitting the form
> and waiting.

## Exactly what to request

**Access**: an OpenSky account with **historical data access** via **Trino**
(``trino.opensky-network.org``; the old Impala shell is deprecated). The challenge's MinIO
bucket comes through Keycloak SSO and grants none of this — it is a separate request. Ask on
the challenge Discord or at challenge@opensky-network.org, citing PRC Data Challenge 2026.

**Table**: ``state_vectors_data4``.

| field | why it is needed |
|---|---|
| ``time`` | the measurement instant — this is the quantity being extracted |
| ``icao24`` | aircraft identity, stable across a movement |
| ``callsign`` | **the join key** to ``CALLSIGN_flt`` |
| ``lat``, ``lon`` | confirm the airport, and locate the stand |
| ``velocity`` | the pushback / movement signal itself |
| ``onground`` | restricts to surface movements |
| ``hour`` | **the partition key** — a query without it scans the whole table and times out |

**Coverage**: 2025-01-01 → 2026-01-01 (the labelled training year), plus 2026-01-01 → 2026-02-01
and 2026-07-01 → 2026-08-01 (the two scored months). Ten airports, fourteen months.

## Do the detection server-side — this decides feasibility

Raw 1 Hz ground state vectors for ten airports over fourteen months is on the order of **10+
billion rows**; even restricted to the hour before each of the 2.43M departures it is ~8
billion. That is not a download, and any plan that starts with "pull the state vectors" fails
here.

Instead, express the detector **as a windowed Trino query that returns one row per flight** —
per ``(icao24, callsign)`` inside an airport bounding box, the last instant the aircraft was
stationary before it began moving. Output is ~2.4M rows rather than billions.

## The cheap kill-switch — run this before asking for anything big

**The training data contains ``BLOCK_TIME_UTC_mvt``.** So the entire idea can be validated on
**one airport-month** without any bulk pull:

1. Pull one airport, one month of 2025.
2. Compute ``offblock_adsb`` for those departures.
3. Compare directly against the known truth:
   ``rmse(offblock_adsb - BLOCK_TIME)`` against ``rmse(AOBT_3_flt - BLOCK_TIME)``.

If the detector does not beat ``AOBT_3_flt`` on a single airport-month, it will not beat it on
fourteen, and the project stops there having cost a day. Only scale after this passes.

For reference, the quantity to beat, measured on training rows (`e22_calibration.py`):
``AOBT_3 - BLOCK`` has median −67 s on ordinary taxis but **+2,758 s** on taxis over an hour —
the hold it misses entirely. A detector that captures the hold should show a flat bias across
all true-taxi bands, which is the diagnostic to look at, not just the aggregate RMSE.

## Hard prerequisites — check these before writing any code

1. **Historical Trino access** — see above. **As of 2026-09-26 unconfirmed**: neither
   `pyopensky`, `traffic` nor `trino` is installed in `.venv`, and there is no
   `~/.config/pyopensky`. This is the long pole; request it before anything else.
2. **Volume sanity check before any bulk pull** — covered by the kill-switch above.
3. **Stand coordinates.** Detecting "stationary at the stand" needs stand geometry, or a
   proxy: the aircraft's own first-seen position on the ground. OurAirports (public domain) is
   already a dependency for airport coordinates but has no stand polygons.
4. **Licensing must be documented** before use — a hard prize-eligibility requirement
   (challenge eligibility rules). OpenSky data is openly licensed for research; record the
   exact terms and attribution in `data/external/*.meta.json` as for Open-Meteo.

## Abandon criteria

Set a hard date. If access is not working **and** one airport-day has not been successfully
processed end to end by then, stop and ship the incremental v4. The realistic downside is
losing the second week for nothing; the upside is the only lever measured to sit on top of the
dominant error population.
