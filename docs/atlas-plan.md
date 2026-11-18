# Atlas plan

How govML gets from the per-dataset pipeline to the atlas described in `PRODUCT.md`. Data first:
the design pass waits until real findings exist.

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
- **Unit records.** If the kept series hold under half of a long table's values (HDB resale prices are
  one row per transaction), the table is unmatched as `low_row_coverage`. It needs an aggregating reader.
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

It reads the theme store and writes `candidates.parquet`/`.json` and `breaks.parquet`/`.json` next
to it. Dataset names come from `catalog.db`. The theme JSON names some datasets only by id.

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
Break   (descriptive annotation, not a corrected family)
  series_id, label, freq, period (first period of the new regime), supF, crit_1pct,
  delta (mean shift in sd units of the transformed series), n
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
   when p_test_adj < 0.05, r_train and r_test have the same sign, and |r_test| >= 0.3.
   *Deviation:* the family is the set of tested pairs, one p each. The spec says pairs × lag
   window, but the lag is chosen on train and never re-searched on test. The spec's own figure
   (F ≈ 35k from about 270 representatives) also counts pairs, not lags.
5. **COVID gate.** The candidate must still pass on one of two variants: Spearman on test, or
   Pearson with every point pair touching 2020-01..2021-12 (2020Q1..2021Q4) removed from both
   windows. Passing means three things: the sign matches r_train (and r_train ex-COVID, for the
   second variant), |r| >= 0.3, and p at most the BH line q × D / F, where D is the number of
   BH discoveries. Both numbers are stored.
6. **Breaks.** These are descriptive. Each representative left after rule 2 gets a sup-F test,
   at any frequency, provided it has at least 20 transformed points. The test looks for one mean
   shift in the deseasoned changes, with 15% trimming and iid residual variance, against
   Andrews' 1% critical value of 12.16. Each series keeps its top break if it clears.
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
- Survivors: 15 (13 comove, 2 lead). BH killed 5,440, sign 2, COVID 4. 40 of 249 series
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

## Phases

1. **Flipper.** Reader registry, both readers, `Series` model and parquet store, a coverage report
   of matched vs unmatched tables with reasons. Delete the 1,654 old run verdicts (back up first).
2. **Theme slice.** Select the housing and cost-of-living datasets, fetch any not yet cached, and
   flip them. Report series count by frequency and span.
3. **Findings engine.** Align pairs by frequency, difference or detrend, test co-movement and
   lead/lag with multiple-testing correction, require the lag to hold out-of-sample, detect
   structural breaks. Output ranked candidates.
4. **Review.** An OpenRouter LLM pass drops same-quantity pairs (a total and its own component) and
   writes finding text and project ideas, labelled verified or brainstormed.
5. **Artifacts.** A weekly batch writes static JSON: map coordinates, theme membership, findings,
   series excerpts, ideas. Cron on the VPS.
6. **Design.** impeccable shaping pass on real findings, then build the static frontend.
7. **Launch.** Public deploy. Done when 20 findings are good enough to share.

## Open

- OpenRouter: key is in `.env`; the review defaults to `anthropic/claude-sonnet-5.5` (override with `--model` or `GOVML_LLM_MODEL`).
- First run results and the choice of where to take v1 next: `docs/findings-v0.md`.
- How the weekly batch hands artifacts to the static host (commit, upload, or deploy hook): decide
  in phase 5.
