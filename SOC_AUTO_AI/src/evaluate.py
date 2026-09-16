"""Evaluate the trained Isolation Forest and write a detailed report."""

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    CONFUSION_MATRIX_PATH,
    EVALUATION_LOG_PATH,
    FEATURES_PATH,
    MODEL_PATH,
    OUTPUTS_DIR,
    RAW_DATA_PATH,
    THRESHOLD_PATH,
)
from src.utils import load_json, load_jsonl, save_json, setup_logger

logger = setup_logger("soc_auto.evaluate", EVALUATION_LOG_PATH)


def calculate_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Calculate classification metrics and confusion-matrix counts."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(true, pred, zero_division=0)),
        "recall": float(recall_score(true, pred, zero_division=0)),
        "f1": float(f1_score(true, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "accuracy": float(accuracy_score(true, pred)),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
        "support": {
            "normal": int((true == 0).sum()),
            "anomaly": int((true == 1).sum()),
        },
    }


def _print_report(result: dict) -> None:
    """Print a human-readable evaluation report."""
    matrix = result["confusion_matrix"]
    print("========================================")
    print("   EVALUATION REPORT - Isolation Forest")
    print("========================================")
    print(f"Threshold:        {result['threshold']:.4f}")
    print(f"Total samples:    {result['total_samples']}")
    print(f"Normal (true):    {result['support']['normal']}")
    print(f"Anomaly (true):   {result['support']['anomaly']}")
    print()
    print("--- Confusion Matrix ---")
    print("                Pred Normal   Pred Anomaly")
    print(f"True Normal     TN={matrix['tn']:<10} FP={matrix['fp']}")
    print(f"True Anomaly    FN={matrix['fn']:<10} TP={matrix['tp']}")
    print()
    print("--- Metrics ---")
    print(f"Precision:        {result['precision']:.4f}")
    print(f"Recall:           {result['recall']:.4f}")
    print(f"F1 Score:         {result['f1']:.4f}")
    print(f"FPR:              {result['fpr']:.4f}")
    print(f"Accuracy:         {result['accuracy']:.4f}")
    print()
    print("--- Đánh giá ---")
    print("[!] Recall quá thấp (< 0.6) - model bỏ sót nhiều anomaly" if result["recall"] < 0.6 else "[OK] Recall tốt (>= 0.6)")
    print("[!] F1 quá thấp (< 0.65) - cần tune threshold" if result["f1"] < 0.65 else "[OK] F1 tốt (>= 0.65)")
    print("[OK] Precision tốt (> 0.7)" if result["precision"] > 0.7 else "[!] Precision thấp (<= 0.7)")
    print("[OK] FPR tốt (< 0.1)" if result["fpr"] < 0.1 else "[!] FPR cao (>= 0.1)")
    print("========================================")


def _save_confusion_matrix(matrix: np.ndarray) -> None:
    """Render and persist the confusion matrix as a PNG."""
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        xlabel="Predicted label",
        ylabel="True label",
        title="Isolation Forest Confusion Matrix",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(figure)


def evaluate_model(data_path=RAW_DATA_PATH) -> dict:
    """Evaluate persisted artifacts and write report, chart, and logs."""
    records = load_jsonl(data_path)
    model = joblib.load(MODEL_PATH)
    threshold = float(load_json(THRESHOLD_PATH)["threshold"])
    features = np.load(FEATURES_PATH)
    labels = np.asarray([int(record["is_anomaly_ground_truth"]) for record in records])
    if len(features) != len(labels):
        raise ValueError(f"Feature and label counts differ: {len(features)} != {len(labels)}")
    scores = model.decision_function(features)
    predictions = (scores < threshold).astype(int)
    result = calculate_metrics(labels, predictions)
    result.update({"total_samples": len(labels), "threshold": threshold})
    _save_confusion_matrix(
        np.array([
            [result["confusion_matrix"]["tn"], result["confusion_matrix"]["fp"]],
            [result["confusion_matrix"]["fn"], result["confusion_matrix"]["tp"]],
        ])
    )
    output_path = OUTPUTS_DIR / "evaluation.json"
    save_json(result, output_path)
    _print_report(result)
    logger.info("evaluation written to %s", output_path)
    logger.info("confusion matrix written to %s", CONFUSION_MATRIX_PATH)
    return result


if __name__ == "__main__":
    evaluate_model()
