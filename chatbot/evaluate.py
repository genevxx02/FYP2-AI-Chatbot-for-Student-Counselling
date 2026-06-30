"""
Model evaluation script for the counselling chatbot NLP pipeline.

Evaluates:
  1. Emotion classification (Hugging Face: j-hartmann/emotion-english-distilroberta-base)
  2. Intent classification (TF-IDF + Logistic Regression)

Metrics: Accuracy, Precision, Recall, F1-score (macro, weighted, per-class)

Usage (from chatbot/ directory):
    python evaluate.py
    python evaluate.py --save-report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from nlp.predict import predict_intent
from sentiment import detect_emotion

EVAL_DIR = Path(__file__).resolve().parent / "evaluation"
EMOTION_TEST_PATH = EVAL_DIR / "test_emotions.json"
INTENT_TEST_PATH = EVAL_DIR / "test_intents.json"
RESULTS_DIR = EVAL_DIR / "results"


def load_test_samples(path: Path) -> tuple[list[str], list[str]]:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    texts = [item["text"] for item in data["samples"]]
    labels = [item["label"] for item in data["samples"]]
    return texts, labels


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    precision_per, recall_per, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class = {}
    for idx, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(precision_per[idx]), 4),
            "recall": round(float(recall_per[idx]), 4),
            "f1_score": round(float(f1_per[idx]), 4),
            "support": int(support[idx]),
        }

    return {
        "accuracy": round(float(accuracy), 4),
        "macro": {
            "precision": round(float(precision_macro), 4),
            "recall": round(float(recall_macro), 4),
            "f1_score": round(float(f1_macro), 4),
        },
        "weighted": {
            "precision": round(float(precision_weighted), 4),
            "recall": round(float(recall_weighted), 4),
            "f1_score": round(float(f1_weighted), 4),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
    }


def evaluate_emotion() -> dict:
    texts, y_true = load_test_samples(EMOTION_TEST_PATH)
    y_pred = [detect_emotion(text)["label"] for text in texts]
    labels = sorted(set(y_true))
    metrics = compute_metrics(y_true, y_pred, labels)
    metrics["model"] = "j-hartmann/emotion-english-distilroberta-base"
    metrics["test_samples"] = len(texts)
    metrics["task"] = "Emotion Classification"
    return metrics


def evaluate_intent() -> dict:
    texts, y_true = load_test_samples(INTENT_TEST_PATH)
    y_pred = [predict_intent(text)["tag"] for text in texts]
    labels = sorted(set(y_true))
    metrics = compute_metrics(y_true, y_pred, labels)
    metrics["model"] = "TF-IDF Vectorizer + Logistic Regression"
    metrics["test_samples"] = len(texts)
    metrics["task"] = "Intent Classification"
    return metrics


def format_summary_table(metrics: dict) -> str:
    lines = [
        f"Task: {metrics['task']}",
        f"Model: {metrics['model']}",
        f"Test samples: {metrics['test_samples']}",
        "",
        "Overall Metrics",
        "-" * 55,
        f"{'Metric':<22} {'Value':>10}",
        "-" * 55,
        f"{'Accuracy':<22} {metrics['accuracy']:>10.4f}",
        f"{'Precision (Macro)':<22} {metrics['macro']['precision']:>10.4f}",
        f"{'Recall (Macro)':<22} {metrics['macro']['recall']:>10.4f}",
        f"{'F1-score (Macro)':<22} {metrics['macro']['f1_score']:>10.4f}",
        f"{'Precision (Weighted)':<22} {metrics['weighted']['precision']:>10.4f}",
        f"{'Recall (Weighted)':<22} {metrics['weighted']['recall']:>10.4f}",
        f"{'F1-score (Weighted)':<22} {metrics['weighted']['f1_score']:>10.4f}",
        "",
        "Per-Class Metrics",
        "-" * 70,
        f"{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}",
        "-" * 70,
    ]
    for label, scores in sorted(metrics["per_class"].items()):
        lines.append(
            f"{label:<16} {scores['precision']:>10.4f} "
            f"{scores['recall']:>10.4f} {scores['f1_score']:>10.4f} "
            f"{scores['support']:>10}"
        )
    lines.extend([
        "",
        "Confusion Matrix",
        f"Labels: {', '.join(metrics['confusion_matrix']['labels'])}",
    ])
    for row in metrics["confusion_matrix"]["matrix"]:
        lines.append("  " + "  ".join(f"{val:>4}" for val in row))
    lines.extend([
        "",
        "Sklearn Classification Report",
        metrics["classification_report"],
    ])
    return "\n".join(lines)


def chapter5_report(emotion_metrics: dict, intent_metrics: dict) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        "=" * 72,
        "CHAPTER 5 — MODEL EVALUATION RESULTS",
        "AI Student Counselling Chatbot",
        f"Generated: {timestamp}",
        "=" * 72,
        "",
        "5.1 Evaluation Methodology",
        "  - Held-out test datasets independent of training data",
        f"  - Emotion test set: {emotion_metrics['test_samples']} labelled samples",
        f"  - Intent test set: {intent_metrics['test_samples']} labelled samples",
        "  - Metrics: Accuracy, Precision, Recall, F1-score",
        "",
        "5.2 Emotion Classification Results",
        "-" * 72,
        format_summary_table(emotion_metrics),
        "",
        "5.3 Intent Classification Results",
        "-" * 72,
        format_summary_table(intent_metrics),
        "",
        "5.4 Summary Comparison",
        "-" * 72,
        f"{'Task':<28} {'Accuracy':>10} {'F1 (Macro)':>12} {'F1 (Weighted)':>14}",
        "-" * 72,
        f"{'Emotion Classification':<28} {emotion_metrics['accuracy']:>10.4f} "
        f"{emotion_metrics['macro']['f1_score']:>12.4f} "
        f"{emotion_metrics['weighted']['f1_score']:>14.4f}",
        f"{'Intent Classification':<28} {intent_metrics['accuracy']:>10.4f} "
        f"{intent_metrics['macro']['f1_score']:>12.4f} "
        f"{intent_metrics['weighted']['f1_score']:>14.4f}",
        "=" * 72,
    ]
    return "\n".join(header)


def main():
    parser = argparse.ArgumentParser(description="Evaluate emotion and intent classifiers")
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save JSON and text reports to evaluation/results/",
    )
    args = parser.parse_args()

    print("Evaluating emotion classifier...")
    emotion_metrics = evaluate_emotion()
    print("Evaluating intent classifier...")
    intent_metrics = evaluate_intent()

    report = chapter5_report(emotion_metrics, intent_metrics)
    print()
    print(report)

    if args.save_report:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = RESULTS_DIR / f"evaluation_{stamp}.json"
        txt_path = RESULTS_DIR / f"evaluation_{stamp}.txt"

        payload = {
            "generated_at": datetime.now().isoformat(),
            "emotion_classification": emotion_metrics,
            "intent_classification": intent_metrics,
        }
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        with open(txt_path, "w", encoding="utf-8") as file:
            file.write(report)

        print(f"\nReports saved:")
        print(f"  {json_path}")
        print(f"  {txt_path}")


if __name__ == "__main__":
    main()
