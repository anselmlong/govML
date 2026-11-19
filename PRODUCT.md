# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Curious Singaporeans, civic-tech hobbyists, students and journalists who suspect data.gov.sg
holds more than anyone has looked at, and who would rather read a finding than build a pipeline
to look for one.

## Product Purpose

govML is a **timeline of Singapore in numbers**. It reads all of data.gov.sg as one body of
**series** (a number over time, such as monthly resale prices or quarterly job vacancies), finds the
moments each series changed course, and groups those moments by date. A year page answers: what
changed in Singapore that year, across every number the government publishes, and what probably
caused it.

Every series also has its own page, so a visitor can follow any entry back to the numbers behind it.

## Positioning

- **Catalog-wide, organised by time.** Other tools show one dataset at a time. govML lines up every
  series against the calendar, so a policy or a shock shows up as many numbers turning at once.
- **Automated, and honest about it.** Everything is computed weekly with no editor in the loop. A
  break is shown only when it clears a strict test. The cause of a cluster is the LLM's guess, and
  the site labels it as a guess ("likely") or says "unexplained".

## Operating Context

A public, read-only static site. A weekly batch job on the VPS fetches the catalog, turns tables
into series, detects breaks, clusters them by date, asks an LLM (through OpenRouter) to name each
cluster, and writes JSON artifacts. The frontend is a static build that reads those artifacts.
There is no always-on backend and no compute triggered by visitors. The per-dataset ML pipeline
remains as an operator-only tool.

## Capabilities and Constraints

- **Coverage.** The front door is the timeline, by year, drilling into quarters and months. It
  covers the whole catalog: every table that can become a series. Single-year snapshots, such as
  input-output tables and census cross-tabs, are not series. They wait for later place-based
  analysis.
- **Event pages.** A cluster is several series from several datasets breaking in the same period,
  so no single table can make an event alone. Each break records its direction, sped up or slowed
  down.
- **Series pages.** Every series has a page: its chart, its breaks, its source dataset. Where the
  LLM reviewer agreed across runs, the page also lists other series it moves with.
- **Map.** The semantic map of all datasets is a secondary page, not the front door.
- **Units.** Units differ between rows of one table. Each series carries its own unit, and
  comparisons work on changes, not levels.
- **LLM.** LLM work (naming clusters, reviewing pairs) goes through OpenRouter with a configurable
  model.

## Brand Commitments

The earlier hydrographic-chart identity is retired with the old product. The new identity will be
shaped with impeccable once the timeline has real content to design around. govML never claims to
be an official government surface.

## Evidence on Hand

- `catalog.db` on the VPS: 4,625 datasets from 75 agencies, all embedded. SingStat is 49% of the
  catalog, and 555 datasets are GeoJSON layers.
- The first full run (`docs/findings-v0.md`) flipped 1,053 cached tables into about 48,600 series.
  - Pairs across datasets yielded only 2 findings the reviewer kept consistently. Most candidates
    were one quantity published in several cuts.
  - 133 series show clear breaks. In housing these line up with known events: the 1997 resale
    peak, the 2012 to 2013 cooling measures and a shift in car numbers in early 2018.
  - That result is why the product moved from relationships to a timeline.

## Product Principles

1. **A year page should make someone say "huh, I didn't know that changed then".** Rank clusters
   by how many independent datasets moved, not by how big one number moved.
2. **Never show a break the data cannot defend.** Never present a guessed cause as fact.
3. **Precompute everything.** A visitor never waits for a model.
4. **Every claim links to its numbers.** Year page to cluster to series to source dataset.

## v1 is done when

All of these hold:
- the whole catalog is fetched;
- every year with a cluster has a page with a labelled cause guess;
- every series has a page, and the map page is kept;
- the weekly batch runs on the VPS;
- the site is publicly deployed.
