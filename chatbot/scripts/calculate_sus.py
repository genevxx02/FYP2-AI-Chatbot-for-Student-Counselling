#!/usr/bin/env python3
"""
Calculate System Usability Scale (SUS) scores from participant responses.

Usage:
  python scripts/calculate_sus.py evaluation/usability_responses.json

JSON format — list of participants, each with 10 scores (1–5):
[
  {"participant": "P1", "scores": [4, 2, 4, 1, 5, 2, 4, 1, 5, 2]},
  ...
]

Odd items (1,3,5,7,9): contribution = score - 1
Even items (2,4,6,8,10): contribution = 5 - score
SUS = sum(contributions) * 2.5  (range 0–100)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def sus_score(scores: list[int]) -> float:
    if len(scores) != 10:
        raise ValueError("SUS requires exactly 10 scores (1–5)")
    total = 0
    for i, s in enumerate(scores, start=1):
        if not 1 <= s <= 5:
            raise ValueError(f"Score must be 1–5, got {s} at item {i}")
        if i % 2 == 1:
            total += s - 1
        else:
            total += 5 - s
    return total * 2.5


def interpret(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 68:
        return "Good (above average)"
    if score >= 50:
        return "OK (marginal)"
    return "Poor"


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/calculate_sus.py evaluation/usability_responses.json")
        return 1

    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    scores_list = []

    print("\nSUS Results")
    print("-" * 40)
    for entry in data:
        pid = entry.get("participant", "?")
        s = sus_score(entry["scores"])
        scores_list.append(s)
        print(f"  {pid}: {s:.1f}/100 — {interpret(s)}")

    if scores_list:
        avg = sum(scores_list) / len(scores_list)
        print("-" * 40)
        print(f"  Mean SUS: {avg:.1f}/100 — {interpret(avg)}")
        print(f"  Participants: {len(scores_list)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
