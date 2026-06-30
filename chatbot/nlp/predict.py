"""
TF-IDF + Logistic Regression intent classifier for the counselling chatbot.

Loads model.pkl, vectorizer.pkl, and intents.json from this directory.
"""

import json
import pickle
import random
from pathlib import Path

NLP_DIR = Path(__file__).resolve().parent
INTENTS_PATH = NLP_DIR / "intents.json"
MODEL_PATH = NLP_DIR / "model.pkl"
VECTORIZER_PATH = NLP_DIR / "vectorizer.pkl"

# Minimum confidence to use an intent response instead of emotion fallback
MIN_INTENT_CONFIDENCE = 0.25

_model = None
_vectorizer = None
_intents_data = None
_intent_map = None


def _load_intents():
    global _intents_data, _intent_map
    if _intents_data is None:
        with open(INTENTS_PATH, encoding="utf-8") as file:
            _intents_data = json.load(file)
        _intent_map = {intent["tag"]: intent for intent in _intents_data["intents"]}
    return _intents_data, _intent_map


def _load_model():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                "Intent model not found. Run: python nlp/train.py"
            )
        with open(MODEL_PATH, "rb") as file:
            _model = pickle.load(file)
        with open(VECTORIZER_PATH, "rb") as file:
            _vectorizer = pickle.load(file)
    return _model, _vectorizer


def predict_intent(user_input):
    """
    Classify user message into an intent tag.

    Returns:
        dict with 'tag' (str) and 'confidence' (float 0-1).
    """
    model, vectorizer = _load_model()
    text = (user_input or "").strip()
    if not text:
        return {"tag": None, "confidence": 0.0}

    features = vectorizer.transform([text])
    tag = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    tag_index = list(model.classes_).index(tag)

    return {
        "tag": str(tag),
        "confidence": round(float(probabilities[tag_index]), 4),
    }


def get_intent_response(tag):
    """Return a random template response for the given intent tag, or None."""
    _, intent_map = _load_intents()
    intent = intent_map.get(tag)
    if intent and intent.get("responses"):
        return random.choice(intent["responses"])
    return None


def get_response(user_input):
    """Classify intent and return a response (CLI / backward-compatible helper)."""
    result = predict_intent(user_input)
    if result["tag"] and result["confidence"] >= MIN_INTENT_CONFIDENCE:
        response = get_intent_response(result["tag"])
        if response:
            return response
    return "I'm not sure how to respond, but I'm here to listen."


if __name__ == "__main__":
    print("Chatbot is running (type 'exit' to stop)\n")

    while True:
        msg = input("You: ")
        if msg.lower() == "exit":
            break

        reply = get_response(msg)
        print("Bot:", reply)
