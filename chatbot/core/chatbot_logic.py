from sentiment import detect_emotion, emotion_category
from nlp.predict import predict_intent
from nlp.malay_normaliser import normalise, extract_malay_emotion_signals
from input_processor import process_input
from core.response_engine import generate_response
from session_memory import get_memory_context, update_memory
from core.claude_service import build_compact_context, generate_response as llm_generate
from rag.knowledge_base import retrieve_with_meta

DEFAULT_CRISIS_KEYWORDS = ["suicide", "kill myself", "die", "hurt myself", "self harm"]
DEFAULT_CRISIS_RESPONSE = (
    "I'm really concerned about you. Please contact Talian HEAL 15555 "
    "or Talian Kasih 15999 immediately. You are not alone."
)
DEFAULT_FALLBACK = "Tell me more about how you're feeling."


def _rule_based_fallback(msg, settings, emotion, intent_result, history, session_id):
    reply = generate_response(
        msg,
        emotion,
        intent_result=intent_result,
        history=history,
        session_id=session_id,
    )
    if reply and reply.strip():
        return reply.strip()

    category = emotion_category(emotion)
    if category == "negative":
        return "I'm sorry you're feeling this way. I'm here to listen."
    if category == "positive":
        return "That's great to hear! "
    return settings.get("default_response") or DEFAULT_FALLBACK


def chatbot_response(
    msg,
    settings=None,
    emotion=None,
    intent_result=None,
    history=None,
    session_id=None,
):
    """
    Chatbot response flow:

      0. Normalise Malaysian input
      1. Crisis keywords (original msg — slang can contain crisis signals)
      2. Emotion/intent detection on normalised text
      3. RAG retrieval
      4. Groq LLM (via core/claude_service.py)
      5. Template fallback if LLM unavailable
    """
    settings = settings or {}

    # Normalise Malaysian input
    normalised = normalise(msg)
    malay_signals = extract_malay_emotion_signals(msg)

    # Crisis check (use original msg — slang can contain crisis signals)
    msg_lower = msg.lower()
    crisis_keywords = settings.get("crisis_keywords") or DEFAULT_CRISIS_KEYWORDS
    if any(kw in msg_lower for kw in crisis_keywords):
        return {
            "response": settings.get("crisis_response") or DEFAULT_CRISIS_RESPONSE,
            "source": "crisis",
            "conversation_intent": None,
            "rag_used": False,
            "rag_chunks": 0,
        }

    # Detect emotion/intent from normalised text
    if emotion is None:
        emotion = detect_emotion(normalised)["label"]
    if intent_result is None:
        try:
            intent_result = predict_intent(normalised)
        except (FileNotFoundError, OSError, ValueError):
            intent_result = None

    # RAG retrieval
    rag_context, rag_chunks = retrieve_with_meta(normalised, top_k=3)

    # Build context
    processed = process_input(normalised, emotion, intent_result)
    update_memory(
        session_id,
        processed["sentiment"],
        processed["intent"],
        processed["keywords"],
    )

    context = build_compact_context(
        emotion=processed["emotion"],
        sentiment=processed["sentiment"],
        intent=processed["intent"],
        keywords=processed["keywords"],
        history=history,
        memory_hints=get_memory_context(session_id),
        rag_context=rag_context,
        malay_signals=malay_signals,
    )

    # Groq LLM
    llm_reply = llm_generate(msg, context)

    if llm_reply and llm_reply.strip():
        return {
            "response": llm_reply.strip(),
            "source": "groq",
            "conversation_intent": processed["intent"],
            "rag_used": bool(rag_context),
            "rag_chunks": rag_chunks,
        }

    # Template fallback
    fallback = _rule_based_fallback(
        msg, settings, emotion, intent_result, history, session_id
    )
    return {
        "response": fallback,
        "source": "template",
        "conversation_intent": processed["intent"],
        "rag_used": False,
        "rag_chunks": 0,
    }
