"""Lightweight normalisation for Malaysian English / Manglish / Malay input."""

import re

# Longer phrases first to avoid partial replacements
_MANGlish_MAP = {
    "very stress": "very stressed",
    "stress la": "stressed",
    "penat gila": "extremely tired",
    "tak tahu": "don't know",
    "macam mana": "how",
    "nak cope": "to cope",
}

_MALAY_EMOTION_CUES = {
    "putus asa": "hopeless",
    "penat gila": "exhausted",
    "penat": "tired",
    "sedih": "sadness",
    "risau": "worried",
    "bimbang": "worried",
    "takut": "fear",
    "gelisah": "anxious",
    "marah": "anger",
    "gembira": "joy",
    "stress": "stress",
    "tertekan": "stressed",
}


def normalise(text: str) -> str:
    """Normalise user text for NLP and RAG retrieval."""
    result = re.sub(r"\s+", " ", (text or "").strip().lower())
    for phrase, canonical in sorted(_MANGlish_MAP.items(), key=lambda x: -len(x[0])):
        result = result.replace(phrase, canonical)
    return result


def extract_malay_emotion_signals(text: str) -> list[str]:
    """Detect Malay/Manglish emotion cues in the original message."""
    lower = (text or "").lower()
    seen = set()
    signals = []
    for phrase, label in sorted(_MALAY_EMOTION_CUES.items(), key=lambda x: -len(x[0])):
        if phrase in lower and label not in seen:
            seen.add(label)
            signals.append(label)
    return signals
