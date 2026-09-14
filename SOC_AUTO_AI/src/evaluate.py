"""Evaluate the trained anomaly model on the sample dataset."""

from pathlib import Path

import joblib
import numpy as np

from src.config import MODEL_PATH, OUTPUTS_DIR, RAW_DATA_PATH, THRESHOLD_PATH
from src.features import build_feature_matrix, load_encoders
from src.utils import load_json, load_jsonl, save_json, setup_logger

logger = setup_logger("soc_auto.evaluate")


def calculate_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Calculate precision, recall, F1, and false-positive rate."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    tp = int(((true == 1) & (pred == 1)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def evaluate_model(data_path=RAW_DATA_PATH) -> dict:
    """Evaluate persisted artifacts and write a JSON report."""
    records = load_jsonl(data_path)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(MODEL_PATH.parent / "scaler.pkl")
    threshold = float(load_json(THRESHOLD_PATH)["threshold"])
    matrix, _, _ = build_feature_matrix(
        records,
        scaler=scaler,
        encoders=load_encoders(MODEL_PATH.parent),
        fit=False,
    )
    scores = -model.decision_function(matrix)
    predictions = (scores > threshold).astype(int).tolist()
    labels = [int(record.get("label", record.get("is_anomaly", 0))) for record in records]
    result = calculate_metrics(labels, predictions)
    result.update({"records": len(records), "threshold": threshold})
    output_path = OUTPUTS_DIR / "evaluation.json"
    save_json(result, output_path)
    logger.info("evaluation written to %s", output_path)
    return result


if __name__ == "__main__":
    evaluate_model()
