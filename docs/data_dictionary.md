# Data dictionary — PRC Data Challenge 2026

Column-by-column reference for the challenge parquet files (`Dataset/training_*.parquet`,
`ranking.parquet`, `submitting.parquet`). All 31 columns, the meaning of every coded value, and the
notation used in the runway/stand/aircraft-type fields.

Counts and null rates below are measured over the full training set — **4,167,797 rows, of which
2,085,047 are DEP** — not taken from the challenge documentation. Regenerate them with
`scripts/` helpers or the snippet at the end of this file.

Related: `docs/findings.md` (the measured findings that drive the design), `SKILLS.md`
(stage-by-stage working checklists).

---

## Three conventions to read first

**1. The `_mvt` / `_flt` suffix names the source system.**

- `_mvt` — the airport's own **movement record**: what actually happened on the ground, from airport
  systems.
- `_flt` — the matched **flight-plan record** from EUROCONTROL's Network Manager: what was filed and
  planned.

The two disagree on a small share of rows (ADES 98.59% agreement, aircraft type 99.66%, flight rule
99.93% on train DEP). The organizers state this inconsistency is intentional and unreconciled, so
`src/cleaning.py` encodes each disagreement as a `dq_*` feature rather than "fixing" either side.

**2. Which airport a row belongs to depends on `PHASE_mvt`.**

For `DEP` rows the movement happens at **`ADEP_mvt`**; for `ARR` rows it happens at **`ADES_mvt`**.
`RUNWAY_mvt` and `STAND_mvt` always belong to that *movement* airport, never to the other endpoint.

This is an easy and silent mistake: filtering `ADEP_mvt == "EDDF"` without also filtering
`PHASE_mvt == "DEP"` mixes Frankfurt departures with arrivals *from* Frankfurt, whose runway and stand
values belong to a completely different airport. The signature of having made the error is a runway
list that doesn't match the airport's real layout.

`ADEP_mvt` has 1,581 distinct values precisely because the *other* endpoint of a flight can be
anywhere in the world; only the 10 challenge airports appear on the movement side.

**3. Every timestamp is UTC**, `datetime64[us, UTC]`. Local civil time matters for rush-hour and
holiday effects and must be derived per airport — `src.external_data.attach_holidays` emits
`local_time` / `local_hour` / `local_dow` / `local_date` using the timezones in `src/airports.py`.

---

## The target

| Column | Meaning |
|---|---|
| `TAXITIME_SEC_mvt` | **Prediction target.** Taxi time in seconds. For DEP rows this is taxi-**out**: off-block (pushback) to takeoff. |

It is exactly `MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt` — a 100.000% exact match on train DEP rows.

Because `BLOCK_TIME_UTC_mvt` is **100% blank in `ranking.parquet`** while `MVT_TIME_UTC_mvt` is 100%
populated there, the real task is *"given when the aircraft got airborne, infer when it left the
stand."* This inverts the usual leakage intuition: the actual takeoff time is a fully legitimate
feature, and anything derived from the off-block time is unusable. See `INFERENCE_BLIND` in
`src/cleaning.py`.

---

## All columns

Null % and distinct counts are over all 4,167,797 training rows (both phases).

### Identity and routing

| Column | Null % | Distinct | Meaning |
|---|---|---|---|
| `MVT_ID_mvt` | 0.0000% | 4,167,797 | Unique movement ID — one per row, and the key the submission file joins on |
| `FLIGHT_ID_mvt` | 0.7421% | 3,821,595 | Network Manager flight identifier. Null ⇒ no flight plan was matched |
| `FLIGHT_mvt` | 0.0098% | 74,532 | Commercial flight number, IATA style (`LH982`, `BA916`) |
| `CALLSIGN_flt` | 0.9098% | 73,054 | ICAO **radio callsign** (`DLH8UL`) — what ATC uses on frequency. Identical to `FLIGHT_mvt` on 73.3% of rows; cargo and charter operators typically fly under their flight number, while large scheduled carriers scramble callsigns to avoid confusion between similar-sounding numbers |
| `ADEP_mvt` / `ADES_mvt` | 0.0020% / 0.0029% | 1,581 / 1,568 | **A**erodrome of **DEP**arture / **DES**tination, ICAO 4-letter codes (`EDDF` = Frankfurt) |
| `ADEP_flt` / `ADES_flt` | 0.9098% | 1,461 / 1,438 | Same, as filed in the flight plan |
| `ADES_FILED_flt` | 0.9098% | 1,436 | Originally filed destination. Differs from `ADES_flt` ⇒ **diversion** (`dq_diverted`) |
| `PHASE_mvt` | 0.0000% | 2 | `DEP` (takeoff) or `ARR` (landing) |
| `AIRCRAFT_OPERATOR_flt` | 0.9098% | 682 | The airline, **SHA-256 hashed** for anonymity. Usable as a high-cardinality categorical; not decodable back to an airline name |

### Timestamps

The `OBT` family is aviation shorthand for **O**ff-**B**lock **T**ime — the moment the aircraft is
pushed back from its parking stand. The letter prefix says *whose* version of that time it is, and the
numeric suffix says which trajectory model produced it (`_1` = filed flight plan, `_3` = actual flown
trajectory, reconstructed from ADS-B).

| Column | Expands to | Null % | Meaning |
|---|---|---|---|
| `SCHED_TIME_UTC_mvt` | — | 0.0000% | Published schedule time |
| `LOBT_flt` | **L**ast Off-Block Time | 0.9098% | Most recently updated planned pushback time |
| `IOBT_flt` | **I**nitial Off-Block Time | 0.9098% | First filed pushback time |
| `EOBT_1_flt` | **E**stimated Off-Block Time | 0.9098% | Estimate from the **filed** trajectory |
| `AOBT_3_flt` | **A**ctual Off-Block Time | 0.9098% | Actual pushback from the **flown** ADS-B trajectory |
| `ARVT_1_flt` / `ARVT_3_flt` | **ARV**al **T**ime | 0.9098% | Arrival time — filed vs flown trajectory |
| `BLOCK_TIME_UTC_mvt` | — | 0.0000% (train) / **100% null (ranking)** | The airport's own off-block (DEP) / in-block (ARR) time |
| `MVT_TIME_UTC_mvt` | **MV**emen**T** time | 0.0000% | Actual takeoff (DEP) or landing (ARR) |

`AOBT_3_flt` and `BLOCK_TIME_UTC_mvt` measure nearly the same physical event through two independent
systems, which is why the former is the single most informative surviving field: it is present on
98.52% of ranking DEP rows, and `MVT_TIME_UTC_mvt − AOBT_3_flt` used raw as a prediction scores
RMSE 380.9 s on trainable rows. Its bias is strongly airport-specific (LTFM +296 s, LIRF −118 s), which
a model can correct.

`ARVT_1_flt` / `ARVT_3_flt` describe events *after* the flight departed and are leakage traps for a
taxi-out model.

### Aircraft and ground position

| Column | Null % | Distinct | Meaning |
|---|---|---|---|
| `AIRCRAFT_TYPE_mvt` | 0.0396% | 271 | ICAO aircraft type designator, movement record |
| `AIRCRAFT_TYPE_flt` | 0.9098% | 242 | Same, flight-plan record |
| `RUNWAY_mvt` | 0.0000% | 58 | Runway used for the movement |
| `STAND_mvt` | 0.0015% | 1,923 | Parking position |

**A missing `AIRCRAFT_TYPE_mvt` is not a routine gap** — it is the signature of the pathological
long-taxi regime. 75.2% of DEP rows over 2 h lack an aircraft type, versus 0.07% of rows overall
(~1000× enrichment), almost entirely at LIRF. Those 1,507 rows are a distinct distribution
(mean 6,451 s, median 2,643 s, σ 11,170 s, against 987 s / 432 s for normal rows), and
`ranking.parquet` contains 108 of them (107 LIRF, 1 LSZH) that must still be predicted.

---

## Coded value reference

### `FLIGHT_RULE_mvt` / `FLIGHT_RULE_flt` — flight rules

| Code | Meaning | Count (`_mvt`) | Count (`_flt`) |
|---|---|---|---|
| `I` | IFR — instrument flight rules | 4,161,029 | 4,127,260 |
| `V` | VFR — visual flight rules | 6,079 | 4 |
| `Y` | IFR first, changing to VFR in flight | — | 1,504 |
| `Z` | VFR first, changing to IFR in flight | — | 969 |
| null | | 689 | 38,060 |

The movement record only ever uses `I`/`V`; `Y` and `Z` appear only on the flight-plan side. VFR
departures taxi far less (median 244 s vs 911 s) but are only ~0.1% of rows.

### `FLIGHT_TYPE_flt` — ICAO type of flight

| Code | Meaning | Count | Median DEP taxi |
|---|---|---|---|
| `S` | Scheduled air service | 3,976,983 | 914 s |
| `N` | Non-scheduled air transport | 108,360 | 816 s |
| `X` | Other / not classified | 23,492 | 778 s |
| `G` | General aviation | 19,878 | 727 s |
| `M` | Military | 1,165 | 678 s |

### `WK_TBL_CAT_flt` — wake turbulence category

**W**a**K**e **T**ur**B**u**L**ence **CAT**egory: how much wingtip vortex turbulence the aircraft
leaves behind. It sets the mandatory separation spacing behind the aircraft, so it maps directly onto
departure-queue time — this is one of the cleanest physical drivers in the dataset.

| Code | Meaning | Count | Median DEP taxi |
|---|---|---|---|
| `L` | Light — MTOW below ~7 t | 20,263 | 603 s |
| `M` | Medium | 3,242,067 | 876 s |
| `H` | Heavy — MTOW above ~136 t | 838,013 | 1,089 s |
| `J` | Super ("Jumbo") — Airbus A380 | 29,394 | 1,326 s |
| null | | 38,060 | |

### `MARKET_SEGMENT_flt` — commercial category

| Value | Count |
|---|---|
| Mainline | 2,651,434 |
| Lowcost | 800,577 |
| Regional | 423,167 |
| Cargo | 120,075 |
| Business | 88,968 |
| Non-Scheduled | 29,287 |
| Other | 14,974 |
| Military | 1,396 |
| null | 37,919 |

Note the value set here is `Cargo` / `Business` / `Non-Scheduled`, not the "All-Cargo" /
"Business Aviation" / "Charter" wording used in the challenge documentation.

---

## Notation used inside the coded fields

### Runway identifiers

A runway is named for its **magnetic heading divided by 10**, plus `L` / `C` / `R` to distinguish
parallel strips: `25C` points at roughly 250°, centre; `09R` points at roughly 090°, right.

The same physical strip carries two names 180° apart (09R in one direction is 27L in the other), and
which one is in use is decided by the wind. So the runway field encodes both the airport's layout
*and* that day's operating configuration.

Departure and arrival runways are systematically different at the same airport, so the field partly
encodes the phase. At Frankfurt, departures overwhelmingly use **18, 25C and 07C** (140,022 / 46,022 /
43,208 rows), while arrivals use the outer parallels **25L, 25R, 07R and 07L** (66,443 / 48,942 /
48,513 / 40,343). Runway 18 — the *Startbahn West* — is departure-only and never appears on an arrival
row. A runway list that doesn't match the airport's real operating pattern is a good sign that a filter
landed on the wrong side of the movement.

`RUNWAY_mvt` uses the literal string `"NA"` for unknown, which `src/cleaning.py` converts to a real
null so it does not silently become its own category.

### Stand identifiers

Parking positions use each airport's own local naming (`B41`, `A62B`, `V119`, `F233`) — there is no
cross-airport standard, so stand is only meaningful in combination with the airport. The letter prefix
identifies the pier or apron area, which makes it a strong proxy for distance from stand to runway
threshold.

**The `V` prefix is Frankfurt-specific.** V-stands cover 38.2% of EDDF departures (87,930 rows) and
essentially nothing anywhere else (10 rows at LSZH, 3 at LTFM). They carry a distinctly lighter
traffic mix than other stands — 91.8% Medium wake versus 77.8% elsewhere, with Regional at 35.9% of
movements — and a shorter median taxi (759 s vs 921 s).

> Worth recording, because the appearance is misleading: V-prefixed stands *look* like they occur at
> Frankfurt, Heathrow, CDG and Istanbul at once, which would suggest a provider-side convention rather
> than a real stand name. That appearance is an artefact of convention 2 above — inspecting stands by
> `ADEP_mvt` without filtering `PHASE_mvt == "DEP"` attributes Frankfurt's stands to every airport that
> receives flights from Frankfurt. On DEP rows only, the pattern is cleanly Frankfurt's.

### Aircraft type designators

ICAO type designators, not manufacturer model names. Common codes in this dataset:

| Code | Aircraft | Code | Aircraft |
|---|---|---|---|
| `A320` | Airbus A320 (ceo) | `B738` | Boeing 737-800 |
| `A20N` | Airbus A320neo | `B38M` | Boeing 737 MAX 8 |
| `A321` / `A21N` | A321ceo / A321neo | `B77W` | Boeing 777-300ER |
| `A319` / `A318` | A319 / A318 | `B772` | Boeing 777-200 |
| `A332` / `A333` | A330-200 / A330-300 | `B789` | Boeing 787-9 |
| `A359` | Airbus A350-900 | `E190` / `E170` | Embraer E190 / E170 |
| `A346` | Airbus A340-600 | `E75S` | Embraer E175 (short wing) |
| `A388` | Airbus A380-800 | `BCS3` | Airbus A220-300 (ex-Bombardier CSeries) |

The `N` in `A20N` / `A21N` marks the neo (new engine option) variants; `M` in `B38M` marks the MAX.

---

## Reproducing the numbers in this file

```python
import glob, pandas as pd

d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("Dataset/training_*.parquet"))],
              ignore_index=True)
dep = d[d["PHASE_mvt"] == "DEP"]                      # movement airport is ADEP_mvt on these rows

# null rate and cardinality for every column
pd.DataFrame({"null_pct": d.isna().mean(), "n_distinct": d.nunique()})

# any coded column
d["WK_TBL_CAT_flt"].value_counts(dropna=False)

# median taxi by code, on trainable DEP rows only
from src.cleaning import clean_movements, trainable_mask
c = clean_movements(d, phase="DEP")
c[trainable_mask(c)].groupby("WK_TBL_CAT_flt")["TAXITIME_SEC_mvt"].median()
```

Median taxi figures quoted above are DEP rows with the target inside 0–3,000 s; the anomalous
missing-aircraft-type regime is excluded by `trainable_mask`, which matters for any mean but barely
moves a median.
