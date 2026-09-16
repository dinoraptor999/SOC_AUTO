"""Inference helpers for one security event."""

from pathlib import Path

import joblib

from src.config import BEST_MODEL_PATH, MODEL_PATH, SCALER_PATH, THRESHOLD_PATH
from src.features import build_feature_matrix, load_encoders
from src.utils import load_json


def load_model(
    model_path: Path | None = None,
    scaler_path: Path = SCALER_PATH,
    threshold_path: Path = THRESHOLD_PATH,
) -> tuple[object, object, dict, float]:
    """Load model artifacts used by the API."""
    model_path = model_path or (BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else MODEL_PATH)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoders = load_encoders(model_path.parent)
    threshold_data = load_json(threshold_path)
    return model, scaler, encoders, float(threshold_data["threshold"])


def predict_one(record: dict, artifacts=None) -> dict:
    """Return anomaly classification and positive anomaly score."""
    loaded = artifacts or load_model()
    model, scaler, _, threshold = loaded
    matrix, _, _ = build_feature_matrix(
        [record], scaler=scaler, encoders=loaded[2], fit=False
    )
    anomaly_score = float(-model.decision_function(matrix)[0])
    is_anomaly = int(anomaly_score > threshold)
    return {
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": anomaly_score,
        "model": "IsolationForest",
        "threshold": threshold,
    }
