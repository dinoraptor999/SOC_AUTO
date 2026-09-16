"""Tests for the FastAPI endpoints."""


def test_health_returns_ok(api_client):
    """GET /health returns the service health contract."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "IsolationForest"


def test_predict_accepts_valid_input(api_client, sample_record):
    """POST /predict accepts a valid event and returns a prediction."""
    response = api_client.post("/predict", json=sample_record)
    assert response.status_code == 200
    assert "is_anomaly" in response.json()
    assert "anomaly_score" in response.json()


def test_predict_rejects_missing_field(api_client):
    """POST /predict returns 422 when required fields are missing."""
    response = api_client.post("/predict", json={"src_ip": "10.0.40.1"})
    assert response.status_code == 422


def test_predict_rejects_invalid_type(api_client, sample_record):
    """POST /predict returns 422 when a field has the wrong type."""
    payload = {**sample_record, "src_port": "not-a-port"}
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422