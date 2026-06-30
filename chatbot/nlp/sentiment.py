"""
Hugging Face emotion classification for the student counselling chatbot.

Uses the pre-trained model:
    j-hartmann/emotion-english-distilroberta-base

Labels: anger, disgust, fear, joy, neutral, sadness, surprise
"""

from transformers import pipeline

# Model identifier on Hugging Face Hub (downloaded automatically on first use)
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Lazy-loaded pipeline so the Flask app starts quickly; model loads on first chat message
_emotion_classifier = None

# Emotion groups used by the existing rule-based response logic
NEGATIVE_EMOTIONS = {"sadness", "anger", "fear", "disgust"}
POSITIVE_EMOTIONS = {"joy"}


def _get_classifier():
    """Create or return the cached transformers text-classification pipeline."""
    global _emotion_classifier
    if _emotion_classifier is None:
        # top_k=1 returns only the highest-confidence emotion label
        _emotion_classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            top_k=1,
        )
    return _emotion_classifier


def detect_emotion(text):
    """
    Classify the dominant emotion in user text.

    Args:
        text: User message string.

    Returns:
        dict with:
            - label (str): emotion label, e.g. "joy", "sadness"
            - confidence (float): model score between 0.0 and 1.0
    """
    if not text or not str(text).strip():
        return {"label": "neutral", "confidence": 0.0}

    classifier = _get_classifier()
    results = classifier(str(text).strip(), truncation=True)

    # Single input: [{"label": "...", "score": 0.9}] or [[{...}]] when top_k is set
    top = results
    if isinstance(top, list) and top:
        top = top[0]
    if isinstance(top, list) and top:
        top = top[0]

    if isinstance(top, dict) and "label" in top and "score" in top:
        return {
            "label": top["label"],
            "confidence": round(float(top["score"]), 4),
        }

    return {"label": "neutral", "confidence": 0.0}


def emotion_category(label):
    """
    Map a fine-grained emotion label to the positive/negative/neutral
    categories used by analytics and fallback responses.
    """
    normalized = (label or "").lower()
    if normalized in POSITIVE_EMOTIONS:
        return "positive"
    if normalized in NEGATIVE_EMOTIONS:
        return "negative"
    return "neutral"
