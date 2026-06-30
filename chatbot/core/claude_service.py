"""
LLM service for counselling replies (Groq API).

Kept as claude_service.py for backward-compatible imports; all generation
uses GROQ_API_KEY — not Anthropic/Claude.
"""

import os
import logging
import time
import re

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "llama-3.3-70b-versatile"
DEFAULT_MAX_TOKENS = 350
MAX_HISTORY_TURNS  = 6
MAX_TURN_CHARS     = 250
MAX_RETRIES        = 3
RETRY_DELAY_SEC    = 1.5

SYSTEM_PROMPT = """You are a warm, empathetic student counsellor for Malaysian university students.

UNDERSTANDING USER LANGUAGE:
Students often write in Malaysian English, Manglish, or mixed Malay-English.
You must understand phrases like "I very stress la", "tak tahu macam mana nak cope",
"penat gila with assignments", and sentence-final particles (la, lah, lo, meh, kan).
Treat these as normal student input — decode the meaning and respond to the feeling or problem.

YOUR REPLY LANGUAGE (IMPORTANT):
- Always reply in clear, natural English only.
- Do NOT use Malay words, Manglish, or Malaysian particles in your replies.
- Do NOT add "la", "lah", "kan", "meh", "lo", "tak", "sikit", or similar to your sentences.
- Do NOT mimic the user's slang or code-switching — understand it, but respond in plain English.
- Match a warm, professional counsellor tone (like a university counselling centre).

RULES:
- Reply in 3-6 sentences only.
- Never repeat the same opening phrase twice.
- Include one practical suggestion when relevant.
- No medical diagnoses.
- Be empathetic and supportive."""

# ── Client ────────────────────────────────────────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is not None:
        return _client
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        log.error("GROQ_API_KEY not set")
        return None
    try:
        from groq import Groq
        _client = Groq(api_key=key)
        log.info("Groq client initialised (model=%s)", _get_model())
        return _client
    except ImportError:
        log.error("groq SDK not installed — run: pip install groq")
        return None

def _get_model():
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL)

def _get_max_tokens():
    try:
        return int(os.getenv("GROQ_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    except ValueError:
        return DEFAULT_MAX_TOKENS

# ── Helpers ───────────────────────────────────────────────────────────────────
def _strip(text, limit=None):
    t = re.sub(r"\s+", " ", (text or "").strip())
    return (t[:limit-3].rstrip() + "...") if limit and len(t) > limit else t

def _build_messages(user_message: str, context: dict) -> list:
    """Build message history for Groq (same format as OpenAI/Claude)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history
    for turn in (context.get("history") or [])[-MAX_HISTORY_TURNS:]:
        u = _strip(turn.get("user_msg"), MAX_TURN_CHARS)
        b = _strip(turn.get("bot_reply"), MAX_TURN_CHARS)
        if u:
            messages.append({"role": "user",      "content": u})
        if b:
            messages.append({"role": "assistant", "content": b})

    # Add RAG context if available
    rag_block = ""
    if context.get("rag_context"):
        rag_block = f"\n\n[Relevant resources]\n{context['rag_context']}"

    # NLP metadata tag (malay_signals help interpret user input — not for mimicking)
    tag_parts = [
        f"emotion={context.get('emotion', 'neutral')}",
        f"intent={context.get('intent', 'unknown')}",
    ]
    malay_signals = context.get("malay_signals") or []
    if malay_signals:
        tag_parts.append(f"user_malay_cues={', '.join(malay_signals)}")

    tag = f"[{'; '.join(tag_parts)}]"

    current = _strip(user_message, 600)
    messages.append({
        "role": "user",
        "content": f"{tag}\n{current}{rag_block}"
    })
    return messages

# ── Core function (same name as before — nothing else needs changing) ─────────
def generate_response(user_message: str, context: dict = None) -> str | None:
    """
    Generate counselling reply via Groq.
    Returns reply string or None if unavailable.
    """
    client = _get_client()
    if client is None:
        return None

    messages = _build_messages(user_message, context or {})
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=_get_model(),
                messages=messages,
                max_tokens=_get_max_tokens(),
                temperature=0.75,
            )
            reply = response.choices[0].message.content.strip()
            log.debug("Groq responded (attempt=%d)", attempt)
            return reply or None

        except Exception as exc:
            last_exc = exc
            log.warning("Groq attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC * attempt)

    log.error("Groq failed after %d attempts: %s", MAX_RETRIES, last_exc)
    return None

# ── Status helpers (called by app.py — keep same interface) ───────────────────
def is_configured() -> bool:
    key = os.getenv("GROQ_API_KEY", "").strip()
    try:
        import groq  # noqa: F401
        sdk_ok = True
    except ImportError:
        sdk_ok = False
    return bool(key) and sdk_ok

def get_status(test_api: bool = False) -> dict:
    """Same structure as before so /api/claude-status still works."""
    has_key = bool(os.getenv("GROQ_API_KEY", "").strip())
    try:
        import groq  # noqa: F401
        sdk_ok = True
    except ImportError:
        sdk_ok = False

    configured = has_key and sdk_ok
    status = {
        "llm_provider":    "groq",
        "claude_enabled":  False,
        "groq_enabled":    configured,
        "api_key_set":     has_key,
        "sdk_installed":   sdk_ok,
        "configured":      configured,
        "active":          configured,
        "response_mode":   "groq" if configured else "template_fallback",
        "model":           _get_model(),
        "max_tokens":      _get_max_tokens(),
    }

    if not has_key:
        status["message"] = "Add GROQ_API_KEY to .env and restart."
    elif not sdk_ok:
        status["message"] = "Run: pip install groq"
    else:
        status["message"] = "Groq active — free LLM replies enabled."

    if test_api and configured:
        client = _get_client()
        try:
            r = client.chat.completions.create(
                model=_get_model(),
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=5,
            )
            status["api_test"]       = "ok"
            status["api_test_reply"] = r.choices[0].message.content.strip()
        except Exception as exc:
            status["api_test"]  = "failed"
            status["api_error"] = str(exc)[:200]
            status["active"]    = False

    return status

# ── Legacy alias (build_compact_context is imported by chatbot_logic.py) ──────
def build_compact_context(*, emotion=None, sentiment=None, intent=None,
                           keywords=None, history=None, memory_hints=None,
                           rag_context=None, malay_signals=None) -> dict:
    return {
        "emotion":       emotion or "neutral",
        "sentiment":     sentiment or "neutral",
        "intent":        intent or "unknown",
        "keywords":      keywords or [],
        "history":       history or [],
        "memory":        memory_hints or [],
        "rag_context":   rag_context,
        "malay_signals": malay_signals or [],
    }