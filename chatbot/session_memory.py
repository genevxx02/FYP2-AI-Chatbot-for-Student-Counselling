# session_memory.py — persists summaries to DB, keeps recent turns lean

from collections import Counter
import logging

log = logging.getLogger(__name__)

_store: dict[str, dict] = {}


def get_or_create(session_id: str) -> dict:
    sid = session_id or "anonymous"
    if sid not in _store:
        _store[sid] = {
            "mood":    Counter(),
            "intents": Counter(),
            "topics":  Counter(),
            "turn_count": 0,
            "summary": None,   # Claude-generated summary of older turns
        }
    return _store[sid]


def update(session_id: str, sentiment: str, intent: str, keywords: list):
    store = get_or_create(session_id)
    store["mood"][sentiment]  += 1
    store["intents"][intent]  += 1
    store["turn_count"]       += 1
    for kw in (keywords or []):
        store["topics"][kw]   += 1


def get_hints(session_id: str) -> list[str]:
    """Return compact hint strings injected into Claude context."""
    store = get_or_create(session_id)
    hints = []

    top_mood = store["mood"].most_common(1)
    if top_mood and top_mood[0][1] >= 2:
        hints.append(f"mood_pattern:{top_mood[0][0]}")

    top_intent = store["intents"].most_common(1)
    if top_intent and top_intent[0][1] >= 2:
        hints.append(f"recurring:{top_intent[0][0]}")

    if store.get("summary"):
        hints.append(f"summary:{store['summary']}")

    return hints


def should_summarise(session_id: str, threshold: int = 8) -> bool:
    """True when conversation is long enough to benefit from summarisation."""
    return get_or_create(session_id).get("turn_count", 0) >= threshold


def store_summary(session_id: str, summary: str):
    get_or_create(session_id)["summary"] = summary[:200]


def clear(session_id: str):
    _store.pop(session_id or "anonymous", None)


# ── Backward-compatible aliases used by chatbot_logic / response_engine ───────
def update_memory(session_id, sentiment, intent, keywords):
    update(session_id, sentiment, intent, keywords)


def get_memory_context(session_id):
    return get_hints(session_id)


def clear_session_memory(session_id):
    clear(session_id)


def memory_opener(hints, intent):
    """Brief continuity phrase from memory hints (template fallback)."""
    if not hints:
        return ""
    if any(h.startswith("recurring:") for h in hints):
        if intent in {"emotional_support", "personal_problem", "academic_help"}:
            return "I notice this theme has come up more than once — "
    if any(h.startswith("mood_pattern:negative") for h in hints):
        if intent == "emotional_support":
            return "You've been carrying a lot lately — "
    return ""