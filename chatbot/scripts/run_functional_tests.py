#!/usr/bin/env python3
"""
Functional testing script for the counselling chatbot.
Run from chatbot/ directory: python scripts/run_functional_tests.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

RESULTS_DIR = ROOT / "evaluation" / "results"


def run_tests() -> list[dict]:
    from app import app, db, init_db
    from chatbot_logic import chatbot_response

    results: list[dict] = []

    def record(test_id: str, name: str, passed: bool, detail: str = ""):
        results.append({
            "id": test_id,
            "name": name,
            "passed": passed,
            "detail": detail,
        })
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_id}: {name}" + (f" — {detail}" if detail else ""))

    print("\n=== Unit tests (chatbot_logic) ===\n")

    crisis = chatbot_response("I want to kill myself")
    record(
        "FT-01", "Crisis keyword detection",
        crisis["source"] == "crisis" and "15555" in crisis["response"],
        f"source={crisis['source']}",
    )

    manglish = chatbot_response(
        "I very stress la, penat gila with exams",
        emotion="fear",
        intent_result={"tag": "stress", "confidence": 0.8},
        session_id="test-manglish",
    )
    record(
        "FT-02", "Manglish input processed",
        bool(manglish.get("response")) and manglish["source"] in {"groq", "template"},
        f"source={manglish['source']}, intent={manglish.get('conversation_intent')}",
    )

    greeting = chatbot_response(
        "hello",
        emotion="neutral",
        intent_result={"tag": "greeting", "confidence": 0.9},
        session_id="test-greeting",
    )
    record(
        "FT-03", "Greeting response generated",
        bool(greeting.get("response")),
        f"source={greeting['source']}",
    )

    print("\n=== API tests (Flask test client) ===\n")

    with app.app_context():
        init_db()

    client = app.test_client()

    r = client.get("/")
    record("FT-04", "Homepage loads", r.status_code == 200, f"status={r.status_code}")

    r = client.post("/get", json={"msg": ""})
    record("FT-05", "Empty message rejected", r.status_code == 400, f"status={r.status_code}")

    r = client.post("/get", json={"msg": "I want to hurt myself"})
    data = r.get_json()
    record(
        "FT-06", "Crisis API response",
        r.status_code == 200 and data.get("response_source") == "crisis",
        f"source={data.get('response_source')}",
    )

    r = client.post("/get", json={"msg": "hi, how are you"})
    data = r.get_json()
    record(
        "FT-07", "Normal chat API response",
        r.status_code == 200 and bool(data.get("response")),
        f"emotion={data.get('emotion')}, source={data.get('response_source')}",
    )

    r = client.get("/api/resources")
    data = r.get_json()
    hotlines = data.get("grouped", {}).get("hotline", [])
    record(
        "FT-08", "Crisis resources API",
        r.status_code == 200 and len(hotlines) >= 1,
        f"hotlines={len(hotlines)}",
    )

    r = client.get("/api/rag-status")
    data = r.get_json()
    record(
        "FT-09", "RAG status API",
        r.status_code == 200 and data.get("ready") is not None,
        f"ready={data.get('ready')}, chunks={data.get('chunk_count')}",
    )

    r = client.get("/api/llm-status")
    data = r.get_json()
    record(
        "FT-10", "LLM status API",
        r.status_code == 200 and "groq_enabled" in data,
        f"active={data.get('active')}, provider={data.get('llm_provider')}",
    )

    r = client.get("/api/session")
    record("FT-11", "Session API", r.status_code == 200, f"logged_in={r.get_json().get('logged_in')}")

    r = client.post("/api/login", json={"email": "wrong@test.com", "password": "wrong"})
    record("FT-12", "Invalid login rejected", r.status_code == 401, f"status={r.status_code}")

    r = client.get("/api/dashboard")
    record("FT-13", "Dashboard requires auth", r.status_code == 401, f"status={r.status_code}")

    r = client.post("/reset")
    record("FT-14", "Chat reset API", r.status_code == 200, f"status={r.status_code}")

    r = client.get("/dashboard", follow_redirects=False)
    record(
        "FT-15", "Legacy dashboard redirects to admin",
        r.status_code in (301, 302, 303, 307, 308),
        f"status={r.status_code}, location={r.headers.get('Location', '')}",
    )

    return results


def save_report(results: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    report_lines = [
        "=" * 72,
        "FUNCTIONAL TEST RESULTS — AI Student Counselling Chatbot",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        f"Summary: {passed}/{total} tests passed ({100 * passed / total:.1f}%)",
        "",
        "Test Cases",
        "-" * 72,
    ]
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        report_lines.append(f"{r['id']:<8} [{status}] {r['name']}")
        if r["detail"]:
            report_lines.append(f"         {r['detail']}")
    report_lines.append("")

    txt_path = RESULTS_DIR / f"functional_tests_{ts}.txt"
    json_path = RESULTS_DIR / f"functional_tests_{ts}.json"
    txt_path.write_text("\n".join(report_lines), encoding="utf-8")
    json_path.write_text(
        json.dumps({"summary": {"passed": passed, "total": total}, "tests": results}, indent=2),
        encoding="utf-8",
    )
    return txt_path


def main():
    print("Running functional tests...")
    results = run_tests()
    path = save_report(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\nDone: {passed}/{len(results)} passed")
    print(f"Report saved: {path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
