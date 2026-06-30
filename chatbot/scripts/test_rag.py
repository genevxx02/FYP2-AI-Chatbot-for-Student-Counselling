#!/usr/bin/env python3
"""Quick RAG retrieval test — run from chatbot/ directory."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from rag.knowledge_base import build_index, retrieve

QUERIES = [
    "I don't know how to manage my time",
    "feeling very stressed about exams",
    "where can I get help on campus",
]


def main():
    count = build_index()
    print(f"Index ready ({count} chunks in knowledge base)\n")

    for q in QUERIES:
        result = retrieve(q)
        preview = (result or "(no match)")[:200]
        print(f"Q: {q}\nA: {preview}\n---")


if __name__ == "__main__":
    main()
