"""Tests for model training and persisted artifacts."""

import src.train as train_module
from src.train import train_model


def test_train_runs_on_ten_row_dataset(small_dataset_path, tmp_path, monkeypatch):
    """Training completes without errors for a ten-row dataset."""
    model_dir = tmp_path / "models"
    monkeypatch.setattr(train_module, "MODELS_DIR", model_dir)
    monkeypatch.setattr(train_module, "THRESHOLD_PATH", model_dir / "threshold.json")
    result = train_model(
        data_path=small_dataset_path,
        model_path=model_dir / "isolation_forest.pkl",
    )
    assert result["threshold"] is not None


def test_train_saves_model_in_models_directory(small_dataset_path, tmp_path, monkeypatch):
    """Training persists the Isolation Forest model under models/."""
    model_dir = tmp_path / "models"
    monkeypatch.setattr(train_module, "MODELS_DIR", model_dir)
    monkeypatch.setattr(train_module, "THRESHOLD_PATH", model_dir / "threshold.json")
    model_path = model_dir / "isolation_forest.pkl"
    train_model(data_path=small_dataset_path, model_path=model_path)
    assert model_path.exists()
    assert model_path.parent == model_dir


def test_train_creates_threshold_json(small_dataset_path, tmp_path, monkeypatch):
    """Training writes a JSON threshold artifact."""
    model_dir = tmp_path / "models"
    threshold_path = model_dir / "threshold.json"
    monkeypatch.setattr(train_module, "MODELS_DIR", model_dir)
    monkeypatch.setattr(train_module, "THRESHOLD_PATH", threshold_path)
    train_model(data_path=small_dataset_path, model_path=model_dir / "model.pkl")
    assert threshold_path.exists()
    assert "threshold" in threshold_path.read_text(encoding="utf-8")