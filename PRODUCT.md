# govML Product Notes

govML is an exploratory Singapore open-data ML workbench. It is designed for a
fast loop:

1. Search or browse the semantic map of data.gov.sg datasets.
2. Inspect metadata, sample rows, schema, and ML suitability.
3. Launch the nine-phase pipeline with conservative defaults.
4. Watch logs and stage progress in the runner.
5. Open a self-contained report with model metrics, diagnostics, context, and
   related datasets.

The product favors graceful degradation. If LLM access (OpenAI), research,
embeddings, optional model packages, or catalog data are unavailable, the
pipeline should continue with heuristics whenever possible.

