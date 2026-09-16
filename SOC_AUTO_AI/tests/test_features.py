"""Tests for feature extraction and matrix construction."""

import json

import numpy as np

from src.config import FEATURE_COLUMNS
from src.features import build_feature_matrix, extract_features


def test_extract_features_returns_all_expected_keys(sample_record):
    """Feature extraction returns the complete numeric feature dictionary."""
    result = extract_features(sample_record)
    expected_keys = {
        "src_port", "dst_port", "failed_count", "bytes",
        "is_internal_src", "is_internal_dst", "is_sensitive_port",
        "is_high_port", "is_failed_auth", "bytes_per_failed",
        "failed_ratio", "is_high_failed", "is_scan_port", "is_night_time",
        "is_high_bytes", "is_low_bytes",
    }
    assert set(result) == expected_keys


def test_build_feature_matrix_returns_numpy_array(small_dataset_path):
    """Feature engineering returns a scaled NumPy matrix with the right shape."""
    records = [json.loads(line) for line in small_dataset_path.read_text().splitlines()]
    matrix, scaler, _ = build_feature_matrix(records)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (10, len(FEATURE_COLUMNS))
    assert scaler.n_features_in_ == len(FEATURE_COLUMNS)


def test_extract_features_handles_sample_record(sample_record):
    """A representative record produces the expected derived values."""
    result = extract_features(sample_record)
    assert result["src_port"] == 50000
    assert result["dst_port"] == 443
    assert result["failed_count"] == 0
    assert result["bytes"] == 1200.0
    assert result["failed_ratio"] == 0.0
    assert result["is_sensitive_port"] == 0