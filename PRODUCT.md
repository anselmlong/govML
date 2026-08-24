# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Civic-tech and open-data enthusiasts: hobbyists, independent researchers, and
builders exploring Singapore's open government data for interesting patterns,
who want to try an idea against a real dataset without first setting up their
own scraping, cleaning, and modeling stack.

## Product Purpose

govML turns a data.gov.sg dataset into a trained model and a readable report
in one pass, so someone curious about a dataset can find out in minutes
whether there's a signal worth pursuing, instead of investing in a pipeline
first and finding out later. The fast loop:

1. Search or browse the semantic map of data.gov.sg datasets.
2. Inspect metadata, sample rows, schema, and ML suitability.
3. Launch the nine-phase pipeline with conservative defaults.
4. Watch logs and stage progress in the runner.
5. Open a self-contained report with model metrics, diagnostics, context, and
   related datasets.

## Positioning

Two things a neighboring tool couldn't truthfully claim together:

- **A semantic map as the discovery surface.** Datasets are embedded and laid
  out spatially so someone can browse by meaning and stumble onto a related
  dataset, not just keyword-search a catalog they'd need to already know how
  to query.
- **Zero-setup, one-click pipeline.** From a raw data.gov.sg dataset ID to a
  trained model and self-contained HTML report, with no pandas/sklearn code,
  no environment setup, and no signup, on the user's own machine.

## Operating Context

Runs locally: a FastAPI backend, a Vite/React frontend, and a Python CLI
pipeline, all against a local SQLite catalog (`catalog.db`) built by ingesting
data.gov.sg's dataset API. No user accounts or server-side deployment; the
person running it owns the machine and the process.

## Capabilities and Constraints

- The product favors graceful degradation: if LLM access, research,
  embeddings, optional model packages, or catalog data are unavailable, the
  pipeline and catalog continue with deterministic heuristics rather than
  failing outright.
- Real OpenAI embeddings (`OPENAI_API_KEY`) meaningfully improve the semantic
  map's clustering quality over the built-in local hashed fallback; both
  paths must keep working since the key is optional infrastructure, not a
  requirement to use the tool.
- The catalog currently holds the full data.gov.sg dataset listing (~4,600+
  datasets) and is expected to keep growing as agencies publish more.

## Brand Commitments

govML has its own distinct visual identity, deliberately departing from
data.gov.sg's own design system (a user-confirmed decision, replacing an
earlier constraint that mirrored it). The identity draws on hydrographic-
chart and land-survey drafting: Singapore's own civic history as a nation
defined by reclaimed land and precise surveying (SLA cadastral plots, the
Singapore Strait's charts) is the visual world for a tool whose core
artifact is itself a spatial chart of datasets. Concretely: an ink/parchment
"chart paper" palette with a bearing-red signature accent, a serif
(Fraunces) used with restraint for headings against a civic sans (Public
Sans) for UI text and a monospace (IBM Plex Mono) for coordinates, scores,
and log/metric readouts, and a compass-rose motif that doubles as the map's
orientation control and the app's loading indicator. Copy stays plain and
literal (Operate-mode conventions: search, ask, run) rather than leaning on
nautical wordplay; the identity lives in the visual system, not the labels.
govML uses its own name and mark throughout, with no claim to be an
official government surface.

## Evidence on Hand

- A live local catalog of real data.gov.sg datasets (`catalog.db`,
  `map_coords.json`), ingested and embedded during development, not sample or
  placeholder data.
- Real pipeline runs have been executed end-to-end against real datasets
  (e.g. HDB resale flat prices), producing real trained models and metrics.

## Product Principles

1. Never let a missing API key or package block the user; degrade to a
   heuristic and say so, don't fail closed.
2. The suitability check comes before the compute: tell someone a dataset is
   a poor fit before they wait through a full training run, not after.
3. One dataset ID in, one self-contained report out; the pipeline should not
   require the user to write or debug code.
4. The map is a discovery tool first, a visualization second: favor
   navigability (real clustering signal, clickable density) over decorative
   polish.
