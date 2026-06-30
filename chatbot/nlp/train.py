"""
Train TF-IDF + Logistic Regression intent classifier from intents.json.
Run from the chatbot/ directory: python nlp/train.py
"""

import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

NLP_DIR = Path(__file__).resolve().parent
INTENTS_PATH = NLP_DIR / "intents.json"
MODEL_PATH = NLP_DIR / "model.pkl"
VECTORIZER_PATH = NLP_DIR / "vectorizer.pkl"


def train():
    with open(INTENTS_PATH, encoding="utf-8") as file:
        data = json.load(file)

    patterns = []
    tags = []
    intent_counts = {}

    for intent in data["intents"]:
        tag = intent["tag"]
        count = len(intent["patterns"])
        intent_counts[tag] = count
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            tags.append(tag)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    features = vectorizer.fit_transform(patterns)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(features, tags)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)
    with open(VECTORIZER_PATH, "wb") as file:
        pickle.dump(vectorizer, file)

    print("NLP model trained successfully!")
    print(f"  Model:      {MODEL_PATH}")
    print(f"  Vectorizer: {VECTORIZER_PATH}")
    print(f"  Total patterns: {len(patterns)}")
    print("  Patterns per intent:")
    for tag, count in intent_counts.items():
        print(f"    {tag}: {count}")


if __name__ == "__main__":
    train()
