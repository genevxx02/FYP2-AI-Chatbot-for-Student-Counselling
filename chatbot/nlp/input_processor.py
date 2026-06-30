"""
Lightweight input processing for the counselling chatbot.

Pipeline per message:
  1. Detect conversation intent (7 categories)
  2. Detect sentiment (positive / neutral / negative)
  3. Extract key keywords (max 5)

Uses the existing TF-IDF intent model as a signal; does not replace it.
"""

import re

from nlp.predict import MIN_INTENT_CONFIDENCE
from sentiment import emotion_category

CONVERSATION_INTENTS = (
    "greeting",
    "emotional_support",
    "academic_help",
    "personal_problem",
    "question",
    "casual_chat",
    "unknown",
)

ML_INTENT_MAP = {
    "greeting": "greeting",
    "goodbye": "casual_chat",
    "stress": "emotional_support",
    "anxiety": "emotional_support",
    "depression": "emotional_support",
    "motivation": "emotional_support",
    "academic_help": "academic_help",
}

GREETING_CUES = (
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "hi there", "hello there",
)

QUESTION_CUES = (
    "?", "how do i", "how can i", "how to", "what is", "what are", "why do",
    "why am", "can you", "could you", "should i", "explain", "tell me",
)

CASUAL_CUES = (
    "thanks", "thank you", "lol", "haha", "ok cool", "nice", "great thanks",
    "bye", "goodbye", "see you", "talk later",
)

PERSONAL_CUES = (
    "relationship", "breakup", "boyfriend", "girlfriend", "family", "parents",
    "friend", "lonely", "bully", "bullied", "argument", "fight with",
)

ACADEMIC_CUES = (
    "study", "exam", "assignment", "coursework", "grade", "lecture", "class",
    "homework", "revision", "thesis", "dissertation", "tutorial",
)

EMOTIONAL_CUES = (
    "stress", "anxious", "anxiety", "depressed", "sad", "overwhelm", "worried",
    "panic", "hopeless", "motivation", "burnout", "crying", "upset", "angry",
)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "i", "me", "my", "mine", "you", "your", "yours",
    "we", "our", "they", "them", "their", "it", "its", "this", "that", "these",
    "those", "am", "im", "just", "so", "very", "really", "about", "from", "as",
    "not", "no", "yes", "can", "cant", "don't", "dont", "feel", "feeling",
}


def detect_sentiment(emotion_label):
    """Map fine-grained emotion to positive / neutral / negative."""
    return emotion_category(emotion_label or "neutral")


def extract_keywords(message, limit=5):
    """Extract up to 5 meaningful keywords (nouns/verbs approximated via filtering)."""
    text = (message or "").lower()
    tokens = re.findall(r"[a-z']+", text)
    keywords = []
    seen = set()

    for token in tokens:
        if len(token) < 3 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= limit:
            break

    return keywords


def _score_cues(text, cues):
    return sum(1 for cue in cues if cue in text)


def detect_conversation_intent(message, ml_intent_result=None):
    """
    Choose one conversation intent using rules + existing TF-IDF classifier output.

    Returns:
        dict with 'intent' (str) and 'source' (str)
    """
    text = (message or "").strip().lower()
    if not text:
        return {"intent": "unknown", "source": "empty"}

    if "?" in text or _score_cues(text, QUESTION_CUES) > 0:
        if _score_cues(text, ACADEMIC_CUES) == 0 or text.endswith("?"):
            return {"intent": "question", "source": "question_cue"}

    if _score_cues(text, GREETING_CUES) > 0 and len(text.split()) <= 8:
        return {"intent": "greeting", "source": "greeting_cue"}

    if _score_cues(text, CASUAL_CUES) > 0 and _score_cues(text, EMOTIONAL_CUES) == 0:
        return {"intent": "casual_chat", "source": "casual_cue"}

    ml_tag = (ml_intent_result or {}).get("tag")
    ml_conf = (ml_intent_result or {}).get("confidence", 0.0)

    if ml_tag in ML_INTENT_MAP and ml_conf >= MIN_INTENT_CONFIDENCE:
        return {"intent": ML_INTENT_MAP[ml_tag], "source": "ml_classifier"}

    if _score_cues(text, ACADEMIC_CUES) > 0:
        return {"intent": "academic_help", "source": "keyword"}

    if _score_cues(text, PERSONAL_CUES) > 0:
        return {"intent": "personal_problem", "source": "keyword"}

    if _score_cues(text, EMOTIONAL_CUES) > 0:
        return {"intent": "emotional_support", "source": "keyword"}

    if ml_tag in ML_INTENT_MAP:
        return {"intent": ML_INTENT_MAP[ml_tag], "source": "ml_low_confidence"}

    return {"intent": "unknown", "source": "fallback"}


def process_input(message, emotion_label, ml_intent_result=None):
    """
    Run the full lightweight input pipeline.

    Returns:
        dict with intent, sentiment, keywords, and emotion label.
    """
    intent_info = detect_conversation_intent(message, ml_intent_result)
    sentiment = detect_sentiment(emotion_label)
    keywords = extract_keywords(message)

    return {
        "intent": intent_info["intent"],
        "intent_source": intent_info["source"],
        "sentiment": sentiment,
        "keywords": keywords,
        "emotion": (emotion_label or "neutral").lower(),
    }
