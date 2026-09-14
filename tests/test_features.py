"""Tests for event feature extraction and preprocessing artifacts."""

from pathlib import Path

from src.config import RAW_DATA_PATH
from src.features import build_feature_matrix, extract_features, load_encoders
from src.utils import load_jsonl


RECORD = {
    "src_ip": "10.0.0.1", "dst_ip": "8.8.8.8", "src_port": 50000,
    "dst_port": 443, "protocol": "tcp", "event_type": "network",
    "failed_count": 0, "bytes": 1000,
}


def test_extract_features_returns_expected_keys():
    """Feature extraction returns the complete numeric contract."""
    result = extract_features(RECORD)
    assert set(result) == {
        "src_port", "dst_port", "failed_count", "bytes",
        "is_internal_src", "is_internal_dst", "is_sensitive_port",
        "is_high_port", "is_failed_auth", "bytes_per_failed",
    }
    assert result["is_internal_src"] == 1
    assert result["is_sensitive_port"] == 0
    assert result["is_high_port"] == 1
    assert result["bytes_per_failed"] == 1000.0


def test_extract_features_handles_missing_values():
    """Missing optional values receive numeric defaults."""
    result = extract_features({"src_ip": "invalid"})
    assert result["failed_count"] == 0
    assert result["bytes"] == 0.0


def test_build_feature_matrix_on_sample_dataset():
    """The sample records produce a scaled matrix and saved artifacts."""
    records = load_jsonl(RAW_DATA_PATH)
    matrix, scaler, encoders = build_feature_matrix(records)
    assert matrix.shape == (20, 12)
    assert scaler.n_features_in_ == 12
    assert set(encoders) == {"protocol", "event_type"}
    model_dir = Path(__file__).resolve().parents[1] / "models"
    assert (model_dir / "scaler.pkl").exists()
    assert (model_dir / "le_protocol.pkl").exists()
    assert (model_dir / "le_event_type.pkl").exists()
    assert set(load_encoders()) == {"protocol", "event_type"}
