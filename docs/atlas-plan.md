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
- **Cache cap.** Tables that hit the 10,000-row cache cap are flagged in the report as possibly truncated.

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

- OpenRouter key and default model: pending from Anselm.
- How the weekly batch hands artifacts to the static host (commit, upload, or deploy hook): decide
  in phase 5.
