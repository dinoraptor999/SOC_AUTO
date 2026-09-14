"""Train and persist the Isolation Forest anomaly model."""

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.config import MODEL_PATH, MODELS_DIR, RAW_DATA_PATH, THRESHOLD_PATH
from src.features import build_feature_matrix
from src.utils import load_jsonl, save_json, setup_logger

logger = setup_logger("soc_auto.train")


def train_model(data_path=RAW_DATA_PATH, model_path=MODEL_PATH) -> dict:
    """Fit Isolation Forest and persist model, scaler, and threshold."""
    records = load_jsonl(data_path)
    matrix, _, _ = build_feature_matrix(records, fit=True)
    model = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    model.fit(matrix)
    # Isolation Forest assigns lower decision_function values to anomalies.
    anomaly_scores = -model.decision_function(matrix)
    threshold = float(np.percentile(anomaly_scores, 95))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    save_json({"threshold": threshold, "percentile": 95}, THRESHOLD_PATH)
    logger.info(
        "trained Isolation Forest on %d records; percentile-95 threshold=%.6f",
        len(records),
        threshold,
    )
    return {"model_path": str(model_path), "threshold": threshold}


if __name__ == "__main__":
    train_model()
