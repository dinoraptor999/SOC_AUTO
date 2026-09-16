"""Compare anomaly detection models on the prepared feature matrix."""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURES_PATH, MODELS_DIR, OUTPUTS_DIR, RAW_DATA_PATH
from src.utils import load_jsonl


BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
COMPARISON_PATH = OUTPUTS_DIR / "model_comparison.json"


def load_ground_truth(path: Path) -> np.ndarray:
    """Load anomaly ground-truth labels from a JSONL file."""
    records = load_jsonl(path)
    return np.asarray(
        [int(record["is_anomaly_ground_truth"]) for record in records],
        dtype=int,
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate precision, recall, F1, and false-positive rate."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
    }


def evaluate_model(name: str, model: object, features: np.ndarray, labels: np.ndarray) -> dict:
    """Fit one model, predict on the feature matrix, and measure timings."""
    train_started = time.time()
    model.fit(features)
    train_time = time.time() - train_started

    inference_started = time.time()
    if isinstance(model, LocalOutlierFactor):
        predictions = (model.negative_outlier_factor_ < model.offset_).astype(int)
    else:
        predictions = (model.predict(features) == -1).astype(int)
    inference_time = time.time() - inference_started

    return {
        "model": name,
        **calculate_metrics(labels, predictions),
        "train_time": train_time,
        "inference_time": inference_time,
        "time": train_time + inference_time,
    }


def main() -> None:
    """Compare configured models, persist the winner, and print a report."""
    features = np.load(FEATURES_PATH)
    labels = load_ground_truth(RAW_DATA_PATH)
    if len(features) != len(labels):
        raise ValueError(f"Feature and label counts differ: {len(features)} != {len(labels)}")

    candidates = [
        (
            "IF contamination=0.1",
            IsolationForest(contamination=0.1, n_estimators=100, random_state=42, n_jobs=-1),
        ),
        (
            "IF contamination=0.3",
            IsolationForest(contamination=0.3, n_estimators=200, random_state=42, n_jobs=-1),
        ),
        (
            "LOF n_neighbors=20",
            LocalOutlierFactor(n_neighbors=20, contamination=0.3),
        ),
    ]
    results = [evaluate_model(name, model, features, labels) for name, model in candidates]
    best_index = max(range(len(results)), key=lambda index: results[index]["f1"])
    best_name, best_model = candidates[best_index]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    comparison = {"best_model": best_name, "results": results}
    with COMPARISON_PATH.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)
        file.write("\n")

    print("Model                          Precision  Recall  F1     FPR    Time(s)")
    print("----------------------------------------------------------------------")
    for result in results:
        print(
            f"{result['model']:<30} {result['precision']:<10.2f} "
            f"{result['recall']:<7.2f} {result['f1']:<6.2f} "
            f"{result['fpr']:<6.2f} {result['time']:.4f}"
        )
    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {BEST_MODEL_PATH}")
    print(f"Saved comparison to: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()