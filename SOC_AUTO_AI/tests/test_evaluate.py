"""Tests for model evaluation and report artifacts."""

import json

import src.evaluate as evaluate_module
import src.train as train_module
from src.features import build_feature_matrix, save_features
from src.train import train_model


def _prepare_evaluation_artifacts(small_dataset_path, tmp_path, monkeypatch):
    """Train temporary artifacts and point evaluation at them."""
    model_dir = tmp_path / "models"
    threshold_path = model_dir / "threshold.json"
    monkeypatch.setattr(train_module, "MODELS_DIR", model_dir)
    monkeypatch.setattr(train_module, "THRESHOLD_PATH", threshold_path)
    model_path = model_dir / "isolation_forest.pkl"
    train_model(data_path=small_dataset_path, model_path=model_path)
    records = [json.loads(line) for line in small_dataset_path.read_text().splitlines()]
    features, _, _ = build_feature_matrix(records)
    features_path = tmp_path / "processed" / "features.npy"
    save_features(features, features_path)
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(evaluate_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(evaluate_module, "FEATURES_PATH", features_path)
    monkeypatch.setattr(evaluate_module, "THRESHOLD_PATH", threshold_path)
    monkeypatch.setattr(evaluate_module, "OUTPUTS_DIR", output_dir)
    monkeypatch.setattr(evaluate_module, "CONFUSION_MATRIX_PATH", output_dir / "confusion_matrix.png")
    return output_dir


def test_evaluate_runs_without_error(small_dataset_path, tmp_path, monkeypatch):
    """Evaluation completes and returns a metrics dictionary."""
    _prepare_evaluation_artifacts(small_dataset_path, tmp_path, monkeypatch)
    result = evaluate_module.evaluate_model(data_path=small_dataset_path)
    assert isinstance(result, dict)
    assert result["total_samples"] == 10


def test_evaluate_metrics_are_between_zero_and_one(small_dataset_path, tmp_path, monkeypatch):
    """All scalar evaluation metrics stay within the inclusive unit interval."""
    _prepare_evaluation_artifacts(small_dataset_path, tmp_path, monkeypatch)
    result = evaluate_module.evaluate_model(data_path=small_dataset_path)
    for metric in ("precision", "recall", "f1", "fpr", "accuracy"):
        assert 0.0 <= result[metric] <= 1.0


def test_evaluate_creates_evaluation_json(small_dataset_path, tmp_path, monkeypatch):
    """Evaluation writes outputs/evaluation.json."""
    output_dir = _prepare_evaluation_artifacts(small_dataset_path, tmp_path, monkeypatch)
    evaluate_module.evaluate_model(data_path=small_dataset_path)
    assert (output_dir / "evaluation.json").exists()