"""Train and persist the Isolation Forest anomaly model."""

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.config import (
    MODEL_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    RAW_DATA_PATH,
    THRESHOLD_PATH,
)
from src.features import build_feature_matrix
from src.utils import load_jsonl, save_json, setup_logger

logger = setup_logger("soc_auto.train", OUTPUTS_DIR / "logs" / "train.log")


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Calculate precision, recall, and F1 without adding another dependency."""
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _ground_truth(records: list[dict]) -> np.ndarray:
    """Read anomaly labels while preserving the project's existing label fields."""
    return np.asarray(
        [int(record.get("is_anomaly_ground_truth", record.get("label", 0))) for record in records],
        dtype=int,
    )


def train_model(
    data_path=RAW_DATA_PATH,
    model_path=MODEL_PATH,
    contamination: float = 0.3,
    n_estimators: int = 200,
) -> dict:
    """Fit Isolation Forest and persist model, scaler, and threshold."""
    records = load_jsonl(data_path)
    matrix, _, _ = build_feature_matrix(records, fit=True)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(matrix)
    # Isolation Forest assigns lower decision_function values to anomalies.
    anomaly_scores = -model.decision_function(matrix)
    threshold = float(np.percentile(anomaly_scores, 95))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    save_json({"threshold": threshold, "percentile": 95}, THRESHOLD_PATH)
    logger.info(
        "trained Isolation Forest on %d records; contamination=%.2f; "
        "n_estimators=%d; percentile-95 threshold=%.6f",
        len(records),
        contamination,
        n_estimators,
        threshold,
    )
    return {
        "model_path": str(model_path),
        "threshold": threshold,
        "contamination": contamination,
    }


def train_with_contamination_tuning(
    data_path=RAW_DATA_PATH,
    model_path=MODEL_PATH,
) -> dict:
    """Tune contamination by F1, then persist the best Isolation Forest model."""
    contaminations = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    records = load_jsonl(data_path)
    matrix, _, _ = build_feature_matrix(records, fit=True)
    labels = _ground_truth(records)
    candidates = []

    logger.info("starting contamination tuning on %d records", len(records))
    print("Contamination | Threshold | Precision | Recall | F1")
    for contamination in contaminations:
        model = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(matrix)
        anomaly_scores = -model.decision_function(matrix)
        threshold = float(np.percentile(anomaly_scores, 95))
        predictions = (anomaly_scores > threshold).astype(int)
        metrics = _metrics(labels, predictions)
        candidate = {
            "contamination": contamination,
            "threshold": threshold,
            **metrics,
        }
        candidates.append(candidate)
        logger.info(
            "contamination=%.2f threshold=%.6f precision=%.4f recall=%.4f f1=%.4f",
            contamination,
            threshold,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
        print(
            f"{contamination:13.2f} | {threshold:9.6f} | "
            f"{metrics['precision']:9.4f} | {metrics['recall']:6.4f} | "
            f"{metrics['f1']:4.4f}"
        )

    best = max(candidates, key=lambda candidate: candidate["f1"])
    final_model = IsolationForest(
        contamination=best["contamination"],
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(matrix)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    save_json(
        {
            "threshold": best["threshold"],
            "percentile": 95,
            "contamination": best["contamination"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
        },
        THRESHOLD_PATH,
    )
    logger.info("selected contamination=%.2f with f1=%.4f", best["contamination"], best["f1"])
    return {"model_path": str(model_path), **best}


if __name__ == "__main__":
    train_with_contamination_tuning()
