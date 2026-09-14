"""Tests for the FastAPI endpoints."""

from fastapi.testclient import TestClient

from src.app import app
from src.train import train_model


def test_health():
    """Health endpoint reports liveness without model artifacts."""
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_predict():
    """Prediction endpoint validates and returns the model contract."""
    train_model()
    payload = {
        "src_ip": "10.0.0.10", "dst_ip": "10.0.0.20", "src_port": 51001,
        "dst_port": 443, "protocol": "tcp", "event_type": "network",
        "failed_count": 0, "bytes": 1200,
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "anomaly_score" in response.json()


def test_predict_validates_required_fields():
    """Prediction rejects payloads missing required event fields."""
    with TestClient(app) as client:
        response = client.post("/predict", json={"src_ip": "10.0.0.1"})
    assert response.status_code == 422
