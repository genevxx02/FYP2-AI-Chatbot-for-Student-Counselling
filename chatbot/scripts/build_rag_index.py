#!/usr/bin/env python3
"""Build or rebuild the active RAG index from knowledge_base/*.md"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from rag.knowledge_base import build_index, get_index_stats  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build counselling RAG index (ChromaDB)")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing index and rebuild from scratch",
    )
    args = parser.parse_args()

    count = build_index(force_rebuild=args.rebuild)
    stats = get_index_stats()
    print(f"RAG index built: {stats['chunks']} chunks from {stats['files']} files")
    return 0 if count is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
