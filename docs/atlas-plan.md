# Atlas plan

How govML gets from the per-dataset pipeline to the timeline described in `PRODUCT.md`. Data
first: the design pass waits until real year pages exist.

## Data shape

Everything downstream reads one model.

```
Series
  series_id   "<dataset_id>:<slug of dimension values>[:<measure>]"   stable across weekly runs
  dataset_id  data.gov.sg dataset id
  label       human label, e.g. "Merchandise Exports: Asia"
  dims        {dimension: value}, e.g. {"region": "Asia"}
  measure     value column for long tables; None for wide tables
  unit        from the table or metadata, e.g. "Million Dollars"; None when unknown
  freq        "M" | "Q" | "H" | "A"
  points      period -> float, ascending, no duplicate periods
```

Stored as two parquet files: `series_meta.parquet` (one row per series) and
`series_points.parquet` (`series_id, period, value`).

Tables reach a series through a **reader registry**: an ordered list of `(detect, read)` pairs,
one per table shape. The first reader whose `detect` matches turns the table into series; a table
no reader matches is recorded with its reason, never silently dropped.

| reader | shape | example |
|---|---|---|
| `wide_time` | rows are series, columns are periods (`2024Sep`, `2024 1Q`, `2024`) | Merchandise Exports By Region, Monthly |
| `long_time` | a time column (`year`, `month`, `quarter`), category columns, value columns | Job Vacancy by Industry, Annual |
| unmatched | single-year snapshots: input-output tables, census cross-tabs | kept for v2 (places) |

### Flipper decisions (phase 1)

Code lives in `src/atlas/`. Run `python -m src.atlas.flip --all` (or `--datasets id1,id2`); it writes
`atlas/series_meta.parquet`, `atlas/series_points.parquet`, `atlas/coverage.json` and `atlas/coverage.md`.

- **Periods** are strings that sort within a frequency and name it: `2024-09` (M), `2024Q3` (Q),
  `2024H2` (H), `2024` (A). `Series.points` is a dict, so a period cannot repeat.
- **Duplicates.** Every non-time, non-numeric column is a dim. Wide tables repeat labels under different
  parents; SingStat indents `DataSeries` by depth, so a repeated label gets a `section` dim holding its
  ancestor chain. Long rows that repeat a period for one dim combination collapse if the values match and
  otherwise skip that series (counted in the report). Nothing is summed or averaged.
- **Unit records.** Tables with one row per event (HDB resale, rentals, CEA transactions) go to the
  `records` reader. That covers two cases: a dim key that repeats within a period, or keys that are
  nearly unique per row with most of them in few periods. The reader emits per-period counts and
  measure medians, overall and per dim of up to 40 values. A long table that still keeps under half
  its values is unmatched as `low_row_coverage`.
- **Units** are read from the row label (`(Million Dollars)`), the value column name (`MillionDollars`),
  or an index base in the dataset name (`(2017 = 100)`). Otherwise `None`.
- **Series shorter than 3 points** are skipped and counted.
- **Cache files.** One file per dataset: the uncapped `_all` if present, else the largest `_<cap>`.
  Files left over are listed in `coverage.json`. A file at its row cap is flagged as possibly truncated.
- **IDs and row order.** A `section` dim or `n` suffix appears only when labels repeat, so inserting a
  row into such a table can rename series that share its label. Known limit on "stable across weekly runs".

### Findings engine (phase 3)

Code: `src/atlas/findings.py` (pipeline, rules, CLI) and `src/atlas/stats.py` (numerics). Run

    python -m src.atlas.findings --store atlas/housing_col --catalog catalog.db --out atlas/housing_col

It reads the theme store and writes `candidates.parquet`/`.json` next to it. Breaks moved to
phase 6 (`src/atlas/breaks.py`). A store whose `breaks.*` files were written by the phase 3
code has the old columns, and `review.py` and `report.py` fail on them until `breaks.py` has
rerun there. Dataset names come from `catalog.db`. The theme JSON names some datasets only by id.

```
Candidate
  kind         "comove" | "lead"            for "lead", x leads y by lag periods
  x, y         series_id; x is always the leader, so lag >= 0
  x_label, y_label
  freq         common frequency after alignment
  lag          periods (0 for comove)
  n_train, n_test, n_eff                     n_eff = Bartlett effective n of the test window
  r_train, r_test, p_test, p_test_adj        adj = BH across the tested family
  r_test_spearman, r_ex_covid, n_ex_covid    robustness (rule 5)
  span         first..last aligned period
  transform    "dlog" | "diff" (or "dlog/diff" when the two differ)
  name_overlap token Jaccard between the two dataset names (0..1)
  score
```

Units vary between rows of one table, so the engine only ever works on changes, never levels.

#### Rules

1. **Representatives.** Each dataset contributes at most K = 3 series per frequency. Headline rows
   come first: empty dims (the records tables' all-rows aggregate), or a dims value starting with
   Total, Overall, All or Whole, or containing "Overall Items". Then come rows with fewer dims
   keys, because `section` children sit one level down. Then table order. In records tables a
   median price or rent ranks before the count, and the count before other medians.
   `median_block` is excluded, since a median of block numbers measures nothing.
   *Deviation:* the spec's fallback, the row with the largest mean |value|, compares levels across
   rows whose units differ, so table order replaces it.
2. **Same quantity.** The spec drops pairs between same-named tables and pairs correlated above
   0.98. On this store that left most survivors as copies of one another. The two tables still
   paired with every third table, so each finding appeared two to four times. Rule 2 therefore
   removes series, not pairs:
   - *Same table.* A table key is the lowercased name with these stripped: frequency words,
     index-base phrases (`(1Q2009 = 100)`, `Base Year 2023 = 100`, `2024 As Base Year`,
     `(BASE QUARTER 1998-Q4 = 100)`), `(Based on ... Date)`, the price basis (`In Chained Volume
     Terms`, `At Current Prices`), a listed breakdown (`By Commodity Section (1-Digit Level)`,
     `By Type Of Property`, `by Market Segment`, `By Locality`, `by Sex`), the URA region
     (`in Core Central Region`) and a leading `Average`/`Total`. Breakdowns are listed rather
     than matched as any `by ...`. In `Resident Households by Monthly Household Employment
     Income` the clause is the subject, and a generic strip merged per-household with
     per-member income. Each key keeps one dataset: the finest frequency first, then one
     with a headline row, then the most points. So 1-, 2- and 3-digit commodity tables, monthly
     and quarterly twins, the four regional transaction tables, the private price index variants
     and town gas average versus total each count once.
   - *Duplicate series.* Two series from different datasets whose changes correlate above 0.98
     over their whole overlap are one quantity. The connected group keeps the series with the
     most points. Season means come from the overlap only. Fitting them on one series' longer
     history made two identical office price indices correlate at 0.96.
   - *Same row.* Pairs between two SITC commodity price indices are dropped. Import, export,
     domestic supply and manufactured products price the same baskets, and domestic supply is
     built from the other two. Also dropped: the same non-headline row (`Core Central Region`)
     in two tables with the same `By ...` breakdown.
   Only pairs from different datasets are considered.
3. **Alignment.** Every series is put on a full period grid first. A change exists only between
   two adjacent periods, so a gap is never bridged. The finer series is averaged down to the
   coarser frequency, and a quarter missing any month stays empty. After transforming, the overlap
   must reach Q 32 / M 60 points. The pair family is M and Q only. Annual series get breaks only,
   because a 30% test window of a median 24-point series holds about 7 points.
4. **Test.** dlog when every level is positive, else first difference. M and Q changes get each
   month's or quarter's mean subtracted, fitted on the train window only. A series is dropped as
   flat when its changes have near-zero variance or are zero in at least half the periods: step
   series such as tariffs and quotas. The overlap is split chronologically 70/30. On train, the
   lag in [-L, L] with the largest |r| is chosen (A 2, Q 8, M 12; ties go to the smaller |lag|).
   That lag is measured once on test. A lagged point pair belongs to train only when both its
   periods are in train, and to test only when both are in test. p_test is a two-sided t test at
   n_eff = n_test / (1 + 2 Σ ρx(j) ρy(j)), with the autocorrelations taken from the train window
   up to lag n_train / 4 and n_eff clamped to [3, n_test]. BH runs at q 0.05. A candidate is kept
   when p_test_adj < 0.05, r_train and r_test have the same sign, |r_test| >= 0.3, and
   |r_train| >= 0.5 × |r_test| (otherwise it is killed as `rule4_episode`: one stretch of the later
   data, not a stable link).
   *Deviation:* the family is the set of tested pairs, one p each. The spec says pairs × lag
   window, but the lag is chosen on train and never re-searched on test. The spec's own figure
   (F ≈ 35k from about 270 representatives) also counts pairs, not lags.
5. **COVID gate.** The candidate must still pass on both of two variants: Spearman on test, and
   Pearson with every point pair touching 2020-01..2022-12 (2020Q1..2022Q4) removed from both
   windows. Passing means three things: the sign matches r_train (and r_train ex-COVID, for the
   second variant), |r| >= 0.3, and p at most the BH line q × D / F, where D is the number of
   BH discoveries. Both numbers are stored. *Changed after the first whole-catalog run:* the
   first version accepted either variant and ended in 2021. With that version, COVID-recovery
   pairs such as aircraft movements against property transactions survived.
6a. **One per dataset pair.** After ranking, only the best candidate for each unordered pair of
   datasets is kept (`rule6_same_dataset_pair`). Arrivals, departures and total from one table
   otherwise repeat a finding three times.
6. **Breaks.** Replaced by phase 6. The first version ran an iid sup-F on representatives
   against Andrews' 1% value of 12.16. On AR(1) series with phi 0.5 and no break it rejects
   26 to 32% of the time, so most of its breaks were serial correlation.
7. **Score.** |r_test| × (1 - name_overlap) × min(1, n_test / 20), descending, with ties broken
   by x, y, lag.
8. **Determinism.** There is no randomness. Pairs are visited in series-id order, every sort is
   stable with explicit tie-breaks, and the JSON is written with sorted keys. Two runs on the same
   store give byte-identical files (sha256-checked, also in the test suite).

#### Numbers from the housing and cost-of-living store

Representatives after rule 2: A 265, M 69, Q 93. Rule 2 dropped 133 for same table and 14 as
duplicates. Cross-dataset M/Q pairs number 30,150 before rule 2 and 10,745 after. The tested
family is F = 5,461. Rule 3 (overlap) dropped 4,394 pairs and flat series 832.

- Median n_test is 22 and median n_eff is 18.2 (M: 45 / 43, Q: 20 / 17).
- BH made 21 discoveries, which puts the cutoff at p = 1.9e-4.
- The smallest detectable |r| at that cutoff is 0.71 at the median n_test and 0.77 at the
  median n_eff.
- Survivors under the first rules: 15 (13 comove, 2 lead). BH killed 5,440, sign 2, COVID 4. The
  tightened rules (episode, the 2020-2022 AND gate, one per dataset pair) leave 3. 40 of 249 series
  clear the break critical value.

The spec's sanity check assumed n_eff around 40. The real median is 18, because most tested
pairs are quarterly. A true r of 0.6 at n_eff 18 has p ≈ 0.008, which no BH family of useful
size admits. Cutting K does not fix this. K = 1 gives F = 819 and a detectable r of 0.70; K = 2
gives 0.76; K = 5 gives 0.74. The binding constraint is the quarterly test window, not F. So K,
the lag window and q are unchanged. Monthly pairs (median n_eff 43) are where real but moderate
relationships can clear.

Across 5,461 tests, 6.3% of raw p fall below 0.05. On synthetic independent AR(0.6) walks the
corrected share is 3%, against 16% with the naive n.

How rule 2 got here. With only the spec's pair drop, 251 candidates survived, most of them
copies. Collapsing tables and duplicate series brought that to 36. Fitting the duplicate check's
season means on the overlap brought it to 34. The average/total strip and the same-row rule
brought it to 23, and the SITC-sibling rule to 15. The BH cutoff tightened from 7.4e-4 to 1.9e-4
while F shrank. Duplicated true positives had been inflating the discovery count D, and that
loosened the cutoff for every pair. So removing series fixes the statistics, not only the
presentation.

What survived: one same-count artefact, the whole-of-Singapore new-sale uncompleted transactions
against uncompleted units sold in RCR, a component of the same count. Plausible relationships:
bond yields against import, export and domestic supply prices, and USD/SGD against food price
indices. Also F&B services against retail sales volumes, where r_train is 0.2 and r_test 0.6, so
the test window's 2020 swing helps even though the ex-COVID r stays at 0.5. Industrial space
prices lead industrial rents by one quarter. New launches move with sales by market segment.
Housing pairs mostly fail on power. Phase 4 still has to judge same-quantity pairs that no name
rule can see.

### Breaks (phase 6)

Code: `src/atlas/breaks.py`. Run

    python -m src.atlas.breaks --store atlas/all --catalog catalog.db --out atlas/all --workers 8
    python -m src.atlas.breaks --simulate --workers 8      # the size and power tables below

It writes `breaks.parquet`/`.json` (one row per break), `break_tests.parquet` (one row per series:
`tested`, `too_short` or `flat`, with the best date, statistic and p even when p is large) and
`breaks_report.json` (the printed counts and the by-year table).

```
Break
  series_id, dataset_id, label
  family        table_key(dataset name), the rule 2 publication key
  freq, period  first period of the new regime
  direction     "accelerated" | "slowed": the mean change per period went up or down
  size          |shift| / within-regime sd of the deseasoned changes
  before, after mean change per period in `transform` units (dlog ~ growth rate)
  transform     "dlog" | "diff"
  statistic, p  sup-Wald and its bootstrap p (999 draws, so p >= 0.001)
  n             changes tested
  duplicate_of  series_id of the kept copy, or None
```

#### What a break is

A break is one shift in the mean of a series' changes. In levels that is a kink: growth sped up or
slowed down and stayed that way. A reader calls this "changed course". Levels after detrending
were rejected for three reasons:
- A jump in levels is a single spike in changes. Most of those spikes are rebasing, a definition
  change or a splice of two tables. With no human check, they would be the bulk of the timeline.
- Units differ between rows, and changes are what phase 3 already compares.
- A one-off shock that recovers, such as the 2008-09 dip or the 2020 drop and rebound, also
  shows up as spikes in changes and not as a mean shift. So the timeline misses V-shaped shocks
  by construction. A separate "shock" kind would need its own outlier test, and that is
  out of scope for v1.

#### The test

1. Take the longest gap-free run of changes (dlog when every level is positive, else first
   difference). Subtract month or quarter means. Skip a series as `too_short` below 30 changes
   or as `flat` when `stats.is_flat` holds.
2. For every break index with at least max(15% of n, 5) changes on each side, compute the Wald
   statistic for a mean shift. The variance is an AR(1)-prewhitened long-run variance of the
   residuals under that break. The statistic is the maximum over break indices. Studentising by
   the long-run variance keeps the statistic close to pivotal in phi.
3. Calibrate it with a wild AR sieve bootstrap: 999 series drawn from an AR(p) fitted to the
   demeaned changes, which is the null of no break. p runs from 1 to the frequency's cap
   (A 1, H 2, Q 4, M 12) and n/10, chosen by AIC. AR(1) gets Kendall's bias correction and roots
   are kept inside 0.97. Residuals keep their period and get a random sign. Every draw is
   deseasoned the same way as the data. p = (1 + #draws >= observed) / 1000. A break is
   reported at p <= 0.01. The rng is seeded from crc32(series_id), so each p depends only on
   its own series, whatever the worker count or the rest of the store.

Variants that failed in `--simulate`-style prototypes:
- The phase 3 iid sup-F against 12.16 rejects 26 to 32% at phi 0.5 and 63 to 84% at phi 0.8.
- The HAC statistic against Andrews' 12.16, with no bootstrap, rejects 4 to 25% at n 30 and
  3 to 8% at n 60.
- The sieve fitted to residuals under the alternative rejects 5 to 17% at n 30 with phi 0.5 to
  0.8, because a sup-chosen break soaks up persistence.
- The iid statistic with a null-fitted sieve holds size but loses most power, because a real
  break inflates phi under the null.
- An iid (not wild) residual resample rejects 5 to 8% when the first third of the series has 3x
  the sd. The wild draw holds 0.8 to 1.2% there.

#### Size and power

Rejection rate at p <= 0.01 on AR(1) series with unit innovations. Size rows use 2,000 series per
cell and power rows use 400. Shifts are in marginal sd, at the middle. The ±2 column is the share of
detected breaks dated within 2 periods of the truth.

| n | size phi 0 / 0.5 / 0.8 | size, monthly style | 1 sd power | 2 sd power | 2 sd dated within ±2 |
|---|---|---|---|---|---|
| 15 | .007 / .010 / .011 | | .02 / .01 / .03 | .09 / .05 / .06 | 1.0 / 1.0 / 1.0 |
| 20 | .009 / .005 / .010 | | .06 / .01 / .01 | .16 / .03 / .03 | .92 / 1.0 / 1.0 |
| 25 | .008 / .005 / .005 | | .10 / .01 / .00 | .34 / .04 / .02 | .96 / .88 / .86 |
| 30 | .011 / .005 / .007 | | .16 / .02 / .02 | .53 / .05 / .02 | .96 / .83 / 1.0 |
| 60 | .010 / .014 / .010 | .004 / .005 / .004 | .63 / .10 / .02 | 1.0 / .39 / .05 | .95 / .81 / .78 |
| 150 | .014 / .009 / .009 | .012 / .003 / .005 | 1.0 / .57 / .08 | 1.0 / .99 / .34 | .95 / .79 / .70 |
| 300 | .011 / .009 / .011 | .006 / .006 / .007 | 1.0 / .95 / .28 | 1.0 / 1.0 / .90 | .95 / .80 / .65 |

The monthly-style column fits 12 season means and AR orders up to 12. A shift planted a quarter
of the way in has a little less power from n 60 up (2 sd: .99 / .25 / .03 at n 60). At n 15 it is undetectable, because the
minimum segment is 5. MA(1) changes with theta -0.5 to -0.95, which are trend-stationary
levels, reject at most 0.5%.

Power at phi 0.8 looks low, but it is at the ceiling. A test told the true phi reaches 0.08 at
n 150 and 0.27 at n 300 for 1 sd with this statistic. A 1 sd shift at phi 0.8 is a third of a
long-run sd. The iid-statistic version of that oracle reaches 0.19 and 0.44, but no test can
match it without knowing phi. Below 30 changes, power is under 0.35 even for a 2 sd shift. On
the real store, series with 15 to 29 changes broke at 0.8 to 1.3%, which is the null rate.
Series with 30 or more broke at 2.8%. That is why `min_changes` is 30.

#### One break

v1 reports at most one break per series. A second break buys no power on these series. On the
HDB resale price index, the best 5-regime fit still leaves residual phi 0.41, against 0.68 with
no break. On the private residential index the figures are 0.54 against 0.69. The swings are
persistence, not clean regimes, and an AR null explains them with or without more breaks.

#### De-duplication

All series are tested, not only representatives. Then breaks are de-duplicated, not series:
comparing every pair in the store would mean about 10^9 pairs. Two breaks are one quantity when
both conditions hold:
- their dates are within a year of each other;
- their changes correlate above 0.98, compared at the coarser of the two frequencies, over at
  least 10 shared periods, with season means fitted on the overlap (rule 2's `full_span_r`).

Each connected group keeps the finest frequency, then the most points, then the smallest id.
The others carry `duplicate_of`. Different cuts of one table that are not copies stay apart:
the 9 age bands of one qualification table that broke in 2001 are 9 breaks. Phase 7 should count families, not
series or datasets. Even `family` overcounts, because `table_key` does not merge male, female
and total versions of one labour-force table.

#### Numbers from the whole-catalog store

The store has 48,637 series from 1,053 tables.
- **Tested.** 12,692 series (A 5,218, M 4,101, Q 3,373). Skipped 35,945: 35,329 too short,
  616 flat.
- **Breaks.** 357 at p <= 0.01 (A 229, M 96, Q 32), a rate of 2.8%. After de-duplication 256,
  from 86 datasets and 65 families.
- **Expected false positives.** At most 127 (1% of tested). At stricter cuts the counts are 263
  at p <= 0.005 and 143 at p <= 0.002.
- **Shape.** 17.4% are dated at a trim edge. 281 slowed and 76 accelerated.
- **Runtime.** 18.6 s for the test on 8 workers (54 s on 1), 23 s wall including reading the
  store.

By year, the de-duplicated count is set against `expected_null`: alpha spread evenly over each
tested series' window of possible break dates. The 15 years with the most distinct datasets:

| year | breaks | deduped | datasets | families | exposed series | expected_null |
|---|---|---|---|---|---|---|
| 1997 | 27 | 17 | 11 | 11 | 7,373 | 2.8 |
| 2001 | 28 | 17 | 8 | 8 | 8,030 | 3.1 |
| 1999 | 17 | 15 | 8 | 8 | 7,411 | 2.7 |
| 2004 | 16 | 12 | 8 | 8 | 8,432 | 3.4 |
| 1995 | 10 | 8 | 8 | 8 | 5,843 | 2.1 |
| 1982 | 11 | 8 | 8 | 5 | 1,006 | 0.3 |
| 2005 | 11 | 8 | 7 | 7 | 8,431 | 3.2 |
| 2014 | 9 | 7 | 7 | 7 | 10,957 | 7.2 |
| 2011 | 12 | 9 | 7 | 6 | 10,175 | 4.5 |
| 2000 | 9 | 8 | 6 | 6 | 7,682 | 2.8 |
| 2012 | 13 | 7 | 6 | 6 | 10,337 | 5.0 |
| 2015 | 7 | 6 | 6 | 6 | 10,754 | 6.6 |
| 1991 | 10 | 8 | 5 | 5 | 4,905 | 1.7 |
| 1989 | 7 | 5 | 5 | 5 | 3,909 | 1.1 |
| 2017 | 6 | 5 | 5 | 5 | 9,609 | 4.4 |

How these peaks read:
- **1997.** 6 of 17 are age bands of one qualification table. The rest are mostly population
  and labour-force age bands, where a cohort moving through a band changes its growth. That is
  demography, not an event. Foreign reserves (1997-04) and merchandise exports (1997-09)
  slowing fit the Asian financial crisis.
- **1999, 2001, 2004.** These are dominated by the same census-style tables: qualification,
  marital status and labour force by age. 2001 also has internet dial-up and mobile
  subscriptions slowing, which is plausibly real saturation. These peaks look like cohort and
  table artefacts.
- **1982.** Six of the eight are SingStat price indices (import, domestic supply, manufactured
  products) slowing together after the 1979-81 inflation. Exposure is only 1,006 series, and 1982 sits
  near the start of their window, so it is plausible but edge-prone.
- **2020.** Retail sales sub-indices "accelerate" from 2020-05/06. That is the rebound after
  the circuit breaker, not a lasting change. 2008-09 barely registers: 2008 has 3 de-duplicated
  breaks against 3.9 expected, and 2009 has 8 against 4.4. Both V-shaped shocks are what this
  design does not see.
- **2011 to 2017.** These years sit at or near `expected_null` and are indistinguishable from
  noise.

#### For phase 7

At p <= 0.01, about a third of the de-duplicated breaks may be noise. In high-exposure years the
null alone supplies 4 to 7 breaks. A fixed "5 series from 3 datasets" bar is met by noise in some
years, so calibrate the cluster bar against `expected_null`. Count families, and merge
sex/total splits first. `break_tests.parquet` holds the best date and p for every tested series,
not only for breaks. That is enough for a pooled test of whether break dates bunch in a year
beyond chance, which can find events that no single series defends.

### Shocks (phase 7)

Code: `src/atlas/shocks.py`. Run

    python -m src.atlas.shocks --store atlas/all --catalog catalog.db --out atlas/all [--breaks atlas/all/breaks.parquet]

It writes `shocks.parquet` (one row per series jump) and `events.json` (the config, the per-level
report with the 40 periods furthest above their null, and the events, ranked by families). Turns
come from a phase 6 `breaks.parquet`, by default the one in `--out`. A phase 3 breaks file is
refused, because it lacks `duplicate_of` and `direction`.

```
Shock    series_id, dataset_id, family, freq
         period      native period of the jump
         slot        quarter ("2020Q2") or, for annual series, year
         z           robust z of the change; direction "up" | "down"
         reversion   share of the jump undone over the next year of changes (None near the end)
Event    period, level ("quarter" | "year")
         families    shocked families; expected = mean of the null count for this period
         p           per-period permutation p; p_familywise against the null's max over periods
         reversion   median over members; top_agency, top_agency_share
         flags       "no_reversion", "one_agency"
         members     family, agency, datasets, shocked/covering series, reversion, the series' jumps
         turns       de-duplicated phase 6 breaks dated in the same calendar year
```

#### The test

1. **Changes.** Each series' longest gap-free run of changes (dlog or diff, as in phase 6), at
   least 40 of them and not flat. Season means come out first.
2. **Local level.** A centred rolling median of the changes, two years each side, comes out next.
   This step is not in the prototype. Without it, a fast-growth stretch sits at z 4 to 6 for
   months on end against a MAD set by the later, flat years. Dial-up and mobile subscriptions did
   this through 2000, which is half of the prototype's 2000Q3. Annual series skip it: a 5-point
   median soaks up a two-year episode, and 8.3% of annual changes then pass z 4, against 2.7%
   without it.
3. **Shock.** |z| >= 4, where z is the residual over 1.4826 × MAD. 3.6% of M/Q changes shock, so
   the tails are heavy, and the null below takes the rate as given.
4. **Family.** `family_key` is the phase 3 `table_key` with two more strips. Population splits
   go (citizen, resident, overall, total, male, female, persons), so the six unemployment tables
   become two families, number and rate. Manpower's `by Industry`, `by Occupation` and
   `by Industry (SSIC 2020) and Occupational Group` cuts also go.

   Then families are merged as copies when any pair of their phase 3 representatives (up to 10
   per dataset and frequency) has changes correlating above 0.9. The changes are taken at the
   level's coarsest frequency, deseasoned, clipped at 3 robust sd and compared over at least 40
   shared changes, with 2020-2022 left out. On the real store, 180 keys become 142 families: the
   HDB resale price index tables, exports with imports and exports by market, total and air
   arrivals, the three non-landed price index tables, the four SITC price indices (rule 2's
   same-row case) and unemployment number with rate.

   Three versions came before this one:
   - One headline per family, unclipped and over 20 changes, missed copies. Arrivals and air
     arrivals by country correlate at 0.999 row by row, but their headlines only at 0.85.
   - Up to 10 representatives per dataset and frequency, unclipped and over 20 changes, chained
     air traffic, mobile subscriptions and private home sales into price-index and arrivals
     groups. Single spikes and short
     overlaps carried r past 0.9.
   - With 2020-2022 kept in, air passengers and visitor arrivals merge, on r 0.985 that comes
     from COVID. An event would then merge its own members.

   A monthly series counts in a quarter when any of its months shocks. A family is shocked in a
   quarter when at least half the series covering that quarter shocked.
5. **Null.** Each family's shocked/not sequence over the quarters it covers is rotated by a random
   offset, seeded, 10,000 times. Families are rotated as units, not series. Series of one table
   jump together. Rotating them apart made "half the family at once" nearly impossible under the
   null: the null total came out at 276 family-quarters against 399 observed, and noise quarters
   such as 2007Q2 and 2023Q1 passed. A rotation keeps each family's own shock count, so the
   observed and null totals now match exactly.
6. **Family-wise bar.** The 99th percentile of the null's maximum count over all quarters, as in
   the prototype. An event is a period whose count reaches it (p_familywise <= 0.01). A
   single-step min-p was tried, so that low-exposure early quarters would not be judged against
   a bar set by 2010s exposure. With 220 quarters it needs on the order of 100,000 reps to
   resolve p_familywise at 0.01. A first version that counted each null draw in its own
   reference set rejected 2.5% at a nominal 1%. The corrected min-p still left 1998Q3 at
   p_familywise 0.68, so it bought nothing, and it was dropped.

#### Artefact flags

Flags label an event. They never remove one.
- `no_reversion` is set when the median member's reversion is below 0.5. A revision or rebasing
  is a step that stays. On planted 12 sd jumps in t(3) noise, a spike that comes back has
  median reversion 1.01 and a step 0.03. Each is on the right side of 0.5 in 97% of 200 trials.
  A recession whose fall lasts a year also steps and stays, so this flag fires on 2008Q4.
- `one_agency` is set when the top agency's share of members beats a one-sided binomial test at
  p < 0.01, against that agency's share of the families covering the period. SingStat is half
  the catalog, so a raw share threshold would flag everything.

#### Size

On synthetic stores of 30 families × 3 quarterly series × 120 quarters, series inside a family
share a t(3) factor and families are independent. Share of stores with any event at the 99% bar,
over 300 seeds each:

| within-family correlation | family rotation (shipped) | per-series rotation (the brief) |
|---|---|---|
| 0.8 | 1.3% | 62% |
| 0.5 | 1.3% | 21% |

The test suite checks 20 seeds at 0.8, and at most 2 may have an event. Runtime on the real
store: the null takes 0.6 s for 10,000 reps, and the whole CLI takes 7 s, mostly reading the
store. The bar is 13 at both 10,000 and 100,000 reps.

#### Annual series

They get their own year-level run with their own null, and nothing is merged into quarters. On
the real store this covers 3,113 series in 46 families. Most families are population by age and
sex, school enrolment, marriages and annulments, and the four SingStat price indices. The
year-level bar is 5 families. The best years are 1982 and 1984, at 3 families each (expected
0.6, p_familywise about 0.6), so the year level has no events. It stays in the batch because it is
cheap and tested against its own null. Annual series cannot be placed in a quarter. Adding them
to the quarter level as four quarters each would let one census table count four times.

#### Numbers from the whole-catalog store

Quarter level: 6,312 M/Q series in 142 families, 50,434 shocks, and a bar of 13 families.

| period | families | expected | p | p_familywise | reversion | flags |
|---|---|---|---|---|---|---|
| 2020Q2 | 18 | 3.5 | 1e-4 | 1e-4 | 0.66 | |

2020Q2 is the COVID circuit breaker, and it is real. Its members:
- visitor arrivals, air traffic, aircraft and transit passengers, vessel calls and arrivals;
- the retail sales index, the F&B services index and GDP;
- town gas use, which went up, and peak power demand;
- short work weeks and active CPF members;
- HDB and private property document lodgements, bank interest rates and state land use;
- two that are likely noise passengers: zakat nisab rates and mobile subscriptions by type.

Phase 6 turns in 2020 attach as secondary: retail sales indices slowing in 2020-05 and
accelerating in 2020-06, and resale flat prices accelerating in 2020-06.

Below the bar, ranked by observed minus expected:
- 2020Q1 has 12 families (expected 3.5, p 1e-4, p_familywise 0.019). This is COVID's onset, and
  it would pass at 95%.
- 2008Q4 has 11 (expected 3.2, p 3e-4, p_familywise 0.07). This is the financial crisis: private
  residential prices and rents, industrial and office rents, short work weeks, the vacancy
  ratio, cargo and foreign reserves. It is flagged `no_reversion` (-0.65), because prices kept
  falling. Its mobile subscriptions member jumps ±15 z between subscription types in 2008-12,
  which is likely a reclassification.
- 2020Q3 has 8 (p 0.018), 2009Q2 has 8 (p 0.007) and 2009Q1 has 7 (p 0.03).
- 1998Q3 has 6 (expected 1.9, p 0.007): unemployment up, private residential and industrial
  prices and rents down. It is the Asian crisis, but it is not defensible family-wise against
  the 2010s exposure.
- 2000Q3 has 4 (expected 2.1, p 0.14): unemployment down 8 z with reversion 0.38, plus dial-up,
  the vacancy ratio and vessels. With season means only, the brief's method, it has 5 (p 0.01),
  adding mobile subscriptions and mobile penetration. There it is flagged both `no_reversion`
  (median -1.16) and `one_agency` (IMDA 60%). Neither method makes it an event.

How the setup moves these:

| variant | families | bar | events | 2020Q1 p_fw | 2008Q4 families, p_fw | 1998Q3 p |
|---|---|---|---|---|---|---|
| shipped | 142 | 13 | 2020Q2 | 0.019 | 11, 0.07 | 0.007 |
| season means only | 142 | 12 | 2020Q2 | 0.012 | 2, 1.0 | 0.001 |
| no copy merge | 180 | 15 | 2020Q2, 2008Q4 | 0.043 | 16, 0.001 | 0.002 |
| season only, no merge | 180 | 13 | 2020Q2 | 0.022 | outside the top 40 | 0.0004 |

2008Q4 is an event only when copies count separately. Without the local level, the property
indices' fall sits inside their own volatile history. No variant makes 1998Q3, 2009Q1 or 2000Q3
an event. The prototype found those three because it rotated series, not families, and counted
copies separately.

#### Open

- The single-quarter unit splits one episode over several quarters: COVID shows as 2020Q1, Q2 and
  Q3. A test over windows of 2 to 3 quarters may lift COVID's onset and 1998 to the bar.
- The family-wise bar is strict by design. At 95%, 2020Q1 would also pass. That is a product
  call.
- Air Traffic Movements' transit passengers and the Aircraft and Transit Passenger table's
  transit rows correlate at 0.95 outside COVID, over fewer than 40 changes. The Air Traffic row
  is past the 10-row cap, and the pair falls under the 40-change overlap, so the two tables stay
  two families in 2020Q2.
- A quarterly table dated by a month per quarter (`1992-03-01`) used to flip as monthly with
  empty months between points, and phase 6 tested none of those series. `flip` now makes such
  series quarterly. The flipper store predates the fix, so `shocks` applies it again on load.
  Phase 6 reads the store as is, so the 12 unemployment series stay `too_short` there and give
  no turns until the store is re-flipped.

## Phases

Phases 1 to 4 ran on 2026-11-19 (results: `docs/findings-v0.md`). The product then moved from
cross-dataset relationships to a timeline (see `PRODUCT.md`). Phases 5 onward are the timeline.

1. **Flipper.** Done.
2. **Theme slice.** Done, and kept as a test fixture.
3. **Findings engine.** Done. Its pair tests now feed only the "moves with" panel on series pages.
4. **Review.** Done. Pairs show only when the reviewer keeps them across repeated runs.
5. **Whole-catalog fetch.** Fetch every tabular dataset not yet cached. Big tables go through the
   bulk `poll-download` endpoint, and the paged API is not used for deep pages. Flip everything and
   report coverage.
6. **Strict breaks.** Done (see "Breaks (phase 6)"). A long-run-variance sup-Wald with a wild AR
   sieve bootstrap holds 1% size on AR(1) series. On the whole store it finds 357 breaks in
   12,692 tested series, 256 after de-duplication.
7. **Shocks and clusters.** Done (see "Shocks (phase 7)"). Each series gets robust-z jumps, and
   table families count once. The null rotates each family's shock sequence. The family-wise bar
   is the 99th percentile of the null's maximum. On the whole store only 2020Q2 (18 families)
   clears the bar of 13. 2020Q1 (12, p_familywise 0.02) and 2008Q4 (11, 0.07) come next. Phase 6
   turns in the same year are listed as secondary.

8. **Cause guesses.** An OpenRouter LLM names each cluster. It is shown as "likely ..." or
   "unexplained", and is never stated as fact. Replies are cached.
9. **Artifacts and weekly batch.** Static JSON for:
   - the year index and year pages;
   - one file per dataset for the series pages;
   - the "moves with" panels;
   - the map.

   Cron on the VPS.
10. **Design.** An impeccable shaping pass on real year pages, then the static frontend: timeline,
    year pages, series pages, map page.
11. **Launch.** Public deploy, which is the v1 done criteria in `PRODUCT.md`.

## Open

- OpenRouter: key is in `.env`; the review defaults to `anthropic/claude-sonnet-5.5` (override with `--model` or `GOVML_LLM_MODEL`).
- First run results and the choice of where to take v1 next: `docs/findings-v0.md`.
- How the weekly batch hands artifacts to the static host (commit, upload, or deploy hook): decide
  in phase 9.
