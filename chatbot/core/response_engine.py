"""
Lightweight intelligent response engine.

Implements structured counselling responses using:
  - conversation intent (7 categories)
  - sentiment branching (positive / neutral / negative)
  - keyword-aware context
  - lightweight session memory
  - template knowledge base (no LLM)
"""

import json
import random
from pathlib import Path

from input_processor import ML_INTENT_MAP, process_input
from session_memory import get_memory_context, memory_opener, update_memory

KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent / "counselling_knowledge.json"

SENTIMENT_FALLBACK_ORDER = {
    "negative": ("negative", "neutral", "positive"),
    "neutral": ("neutral", "negative", "positive"),
    "positive": ("positive", "neutral", "negative"),
}

_knowledge = None
_session_used_responses: dict[str, set[str]] = {}
_session_used_openers: dict[str, set[str]] = {}


def _load_knowledge():
    global _knowledge
    if _knowledge is None:
        with open(KNOWLEDGE_PATH, encoding="utf-8") as file:
            _knowledge = json.load(file)
    return _knowledge


def _map_history_intent(raw_intent):
    if not raw_intent:
        return None
    if raw_intent in ML_INTENT_MAP:
        return ML_INTENT_MAP[raw_intent]
    return raw_intent


def _history_opener(history, intent, session_id=None):
    """Use a varied continuity phrase only when the same topic repeats."""
    if not history:
        return ""

    prior_intents = [
        _map_history_intent(turn.get("conversation_intent") or turn.get("intent"))
        for turn in history
    ]
    if intent not in prior_intents:
        return ""

    openers = [
        "Picking up on that thread — ",
        "Since we were just discussing this — ",
        "To continue from before — ",
    ]
    return _pick_opener(session_id, openers)


def _pick_opener(session_id, openers):
    used = _session_used_openers.setdefault(session_id or "anonymous", set())
    available = [o for o in openers if o not in used]
    if not available:
        used.clear()
        available = list(openers)
    choice = random.choice(available)
    used.add(choice)
    return choice


def _keyword_context(keywords, intent):
    if not keywords:
        return ""

    topic = keywords[0]
    if intent == "academic_help" and topic in {"exam", "exams", "assignment", "study", "deadline"}:
        return f"Regarding {topic}, "
    if intent == "emotional_support" and topic in {"stress", "anxiety", "sad", "worried", "overwhelm"}:
        return f"When {topic} shows up like this, "
    return ""


def _pick_response(session_id, responses):
    if not responses:
        return None

    used = _session_used_responses.setdefault(session_id or "anonymous", set())
    available = [response for response in responses if response not in used]

    if not available:
        used.clear()
        available = list(responses)

    choice = random.choice(available)
    used.add(choice)
    return choice


def _get_response_pool(knowledge, intent, sentiment):
    intent_data = knowledge.get("intents", {}).get(intent, {})
    for key in SENTIMENT_FALLBACK_ORDER.get(sentiment, ("neutral",)):
        pool = intent_data.get(key, [])
        if pool:
            return pool
    return knowledge.get("intents", {}).get("unknown", {}).get("neutral", [])


def _apply_intent_style(intent, sentiment, base, keywords):
    """Apply intent-specific response shaping rules from the system prompt."""
    if intent == "academic_help" and sentiment in {"neutral", "negative"}:
        if "step" not in base.lower() and sentiment == "neutral":
            prefix = "Step one: break the task down. Step two: start with the smallest piece. "
            return prefix + base

    if intent == "question" and not base.lower().startswith(("good question", "direct answer", "here is")):
        return "Short answer first: " + base[0].lower() + base[1:] if base else base

    if intent == "unknown" and "?" not in base:
        return base + " Can you share a bit more detail?"

    if intent == "emotional_support" and sentiment == "negative":
        return base

    if intent == "casual_chat" and sentiment == "positive":
        return base

    if keywords and intent == "personal_problem":
        return base

    return base


def generate_response(message, emotion, intent_result=None, history=None, session_id=None):
    """
    Build a structured counselling reply.

    Pipeline:
      1. process_input() → intent, sentiment, keywords
      2. update session memory
      3. select template by intent + sentiment
      4. apply context openers (history + memory + keywords)
    """
    knowledge = _load_knowledge()
    history = history or []
    session_id = session_id or "anonymous"

    processed = process_input(message, emotion, intent_result)
    intent = processed["intent"]
    sentiment = processed["sentiment"]
    keywords = processed["keywords"]

    update_memory(session_id, sentiment, intent, keywords)
    memory_hints = get_memory_context(session_id)

    pool = _get_response_pool(knowledge, intent, sentiment)
    base = _pick_response(session_id, pool)

    if not base:
        base = (
            "Could you tell me a little more about what you need — "
            "study support, emotional support, or something personal?"
        )

    styled = _apply_intent_style(intent, sentiment, base, keywords)

    opener = memory_opener(memory_hints, intent) or _keyword_context(keywords, intent)
    if not opener and len(history) >= 2:
        opener = _history_opener(history, intent, session_id)

    if opener and not styled.lower().startswith(opener.lower().strip()):
        reply = f"{opener}{styled[0].lower()}{styled[1:]}" if styled else opener
    else:
        reply = styled

    return reply.strip()
