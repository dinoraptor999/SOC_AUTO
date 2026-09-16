"""Tune the Isolation Forest anomaly threshold against labeled events."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "isolation_forest.pkl"
FEATURES_PATH = ROOT / "dataset" / "processed" / "features.npy"
LABELS_PATH = ROOT / "dataset" / "raw" / "sample.jsonl"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"


def load_ground_truth(path: Path) -> np.ndarray:
    """Load ``is_anomaly_ground_truth`` labels from a JSONL file."""
    with path.open("r", encoding="utf-8") as file:
        labels = [int(json.loads(line)["is_anomaly_ground_truth"]) for line in file if line.strip()]
    return np.asarray(labels, dtype=int)


def main() -> None:
    """Evaluate candidate thresholds and persist the one with the best F1."""
    model = joblib.load(MODEL_PATH)
    features = np.load(FEATURES_PATH)
    labels = load_ground_truth(LABELS_PATH)

    if len(features) != len(labels):
        raise ValueError(
            f"Feature and label counts differ: {len(features)} != {len(labels)}"
        )

    scores = model.decision_function(features)
    results = []
    print("Percentile | Threshold | Precision | Recall | F1")
    print("-----------|-----------|-----------|--------|----")
    for percentile in range(70, 99, 2):
        threshold = float(np.percentile(scores, 100 - percentile))
        predictions = (scores < threshold).astype(int)
        precision = float(precision_score(labels, predictions, zero_division=0))
        recall = float(recall_score(labels, predictions, zero_division=0))
        f1 = float(f1_score(labels, predictions, zero_division=0))
        result = {
            "threshold": threshold,
            "percentile": percentile,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        results.append(result)
        print(
            f"{percentile:10d} | {threshold:9.6f} | {precision:9.4f} | "
            f"{recall:6.4f} | {f1:4.4f}"
        )

    best = max(results, key=lambda result: result["f1"])
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with THRESHOLD_PATH.open("w", encoding="utf-8") as file:
        json.dump(best, file, indent=2)
        file.write("\n")

    print("BEST")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()