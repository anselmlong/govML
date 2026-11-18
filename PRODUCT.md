# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Curious Singaporeans, civic-tech hobbyists, students and journalists who suspect data.gov.sg
holds more than anyone has looked at, and who would rather read a finding than build a pipeline
to look for one.

## Product Purpose

govML is an atlas of Singapore in numbers. It reads data.gov.sg as one connected body of
**series** (a number over time, such as monthly resale prices or quarterly CPI), and pre-computes
what relates to what across the catalog:

- which series move together, and which leads which, and by how long;
- where a series broke (COVID, a policy change);
- which datasets share a key and can actually be joined.

A visitor browses findings they could share, not models they would have to judge. Each finding
links to the series behind it and to the project ideas it opens up.

## Positioning

- **Cross-dataset, not per-dataset.** Other tools show one dataset at a time. govML's unit is the
  relationship between datasets, found across the whole catalog.
- **Every finding is earned.** A relationship is shown only after it survives detrending,
  multiple-testing correction and an out-of-sample lag check, and an LLM review has removed pairs
  that are one quantity counted twice. Coincidences stay out.

## Operating Context

A public, read-only static site. A weekly batch job on the VPS ingests the catalog, turns tables
into series, tests relationships, and writes JSON artifacts; the frontend is a static build that
reads them. There is no always-on backend and no visitor-triggered compute. The per-dataset ML
pipeline remains as an operator-only tool.

## Capabilities and Constraints

- The front door is the semantic map of all ~4,600 datasets. v1 lights up one theme, housing plus
  cost of living, and draws verified findings as lines between the datasets they connect.
- Most SingStat tables put time across the columns and need flipping; most other agencies' tables
  are already long-form. Both become series. Single-year snapshots (input-output tables, census
  cross-tabs) are not series and wait for v2's place-based analysis.
- Units differ between rows of one table, so each series carries its own unit and comparisons work
  on changes, not levels.
- Project ideas come in two labelled kinds: verified ones derived from a finding or a tested join,
  shown first, and brainstormed ones, shown separately.
- LLM work (the same-quantity review, idea write-ups) goes through OpenRouter with a configurable
  model.

## Brand Commitments

The earlier hydrographic-chart identity is retired with the old product. The new identity will be
shaped with impeccable once v1 has real findings to design around. govML never claims to be an
official government surface.

## Evidence on Hand

- `catalog.db` on the VPS: 4,625 datasets from 75 agencies, all embedded
  (`text-embedding-3-small`). SingStat is 49% of the catalog; 555 datasets are GeoJSON layers.
- 1,685 datasets cached as parquet. Of those, about 180 SingStat tables are wide time tables and
  about 900 other tables are long-form with a time column.
- The 1,654 per-dataset run verdicts from August 2026 are being deleted: many "good" verdicts were
  trivial (a table's total predicted from its own parts) or nonsense (day-of-week from a prayer
  timetable).

## Product Principles

1. A finding a visitor would share beats ten they would skim. Rank by surprise, cut by rigour.
2. Never show a relationship the data cannot defend; when in doubt, leave it out.
3. Precompute everything. A visitor never waits for a model.
4. The map is for discovery. Clusters, highlighted themes and finding lines carry real signal, not
   decoration.

## v1 is done when

The site is public, the whole-catalog map is live with the housing and cost-of-living theme lit,
at least 20 verified findings are good enough to share personally, and the weekly batch runs on
the VPS.
