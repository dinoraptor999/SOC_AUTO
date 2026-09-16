"""Shared pytest fixtures for the SOC Auto test suite."""

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.app as app_module
import src.features as features_module
import src.train as train_module
from src.config import SCALER_PATH
from src.config import MODELS_DIR as PROJECT_MODELS_DIR
from src.predict import load_model
from src.train import train_model


@pytest.fixture
def sample_record() -> dict:
    """Return one representative security event."""
    return {
        "src_ip": "10.0.40.10",
        "dst_ip": "10.0.30.20",
        "src_port": 50000,
        "dst_port": 443,
        "protocol": "TCP",
        "event_type": "connection_attempt",
        "failed_count": 0,
        "bytes": 1200,
    }


@pytest.fixture
def small_dataset_path(tmp_path: Path) -> Path:
    """Write a reproducible ten-row labeled dataset in a temporary directory."""
    records = []
    for index in range(10):
        anomaly = int(index >= 5)
        records.append({
            "src_ip": f"10.0.40.{index + 1}",
            "dst_ip": f"10.0.30.{index + 1}",
            "src_port": 50000 + index if anomaly else 30000 + index,
            "dst_port": 22 if anomaly else 443,
            "protocol": "TCP",
            "event_type": "authentication_failed" if anomaly else "connection_attempt",
            "failed_count": 10 + index if anomaly else index % 3,
            "bytes": 50 if anomaly else 1200 + index,
            "is_anomaly_ground_truth": anomaly,
        })
    path = tmp_path / "events.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def api_client(tmp_path: Path, small_dataset_path: Path, monkeypatch):
    """Train temporary API artifacts and provide a client with teardown."""
    model_dir = tmp_path / "models"
    model_path = model_dir / "isolation_forest.pkl"
    monkeypatch.setattr(train_module, "MODELS_DIR", model_dir)
    monkeypatch.setattr(train_module, "THRESHOLD_PATH", model_dir / "threshold.json")
    monkeypatch.setattr(features_module, "MODELS_DIR", model_dir)
    train_model(data_path=small_dataset_path, model_path=model_path)
    for encoder_name in ("le_protocol.pkl", "le_event_type.pkl"):
        shutil.copy2(PROJECT_MODELS_DIR / encoder_name, model_dir / encoder_name)
    artifacts = load_model(
        model_path=model_path,
        scaler_path=SCALER_PATH,
        threshold_path=model_dir / "threshold.json",
    )
    monkeypatch.setattr(
        app_module,
        "load_model",
        lambda: artifacts,
    )
    with TestClient(app_module.app) as client:
        yield client
    app_module._artifacts = None