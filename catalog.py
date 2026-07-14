#!/usr/bin/env python
"""Thin CLI wrapper over src.catalog."""

from __future__ import annotations

import argparse
import json

from src import catalog
from src.llm import DEFAULT_EMBED_MODEL


def main() -> int:
    parser = argparse.ArgumentParser(description="govML catalog tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--refresh", action="store_true")
    p_ingest.add_argument("--max-pages", type=int)

    p_embed = sub.add_parser("embed")
    p_embed.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    p_embed.add_argument("--batch-size", type=int, default=64)
    p_embed.add_argument("--limit", type=int)
    p_embed.add_argument("--force", action="store_true")

    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=10)

    p_related = sub.add_parser("related")
    p_related.add_argument("dataset_id")
    p_related.add_argument("--top-k", type=int, default=6)

    sub.add_parser("stats")

    args = parser.parse_args()
    if args.cmd == "ingest":
        catalog.ingest(refresh=args.refresh, max_pages=args.max_pages)
    elif args.cmd == "embed":
        catalog.embed(model=args.model, batch_size=args.batch_size, limit=args.limit, force=args.force)
    elif args.cmd == "search":
        print(json.dumps(catalog.search(args.query, args.top_k), indent=2))
    elif args.cmd == "related":
        print(json.dumps(catalog.related(args.dataset_id, args.top_k), indent=2))
    elif args.cmd == "stats":
        print(json.dumps(catalog.stats(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

