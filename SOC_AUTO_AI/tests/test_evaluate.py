"""Tests for evaluation metrics."""

from src.evaluate import calculate_metrics


def test_calculate_metrics():
    """Metric values are correct for a simple confusion matrix."""
    result = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert result["precision"] == 2 / 3
    assert result["recall"] == 1.0
    assert result["fpr"] == 0.5
