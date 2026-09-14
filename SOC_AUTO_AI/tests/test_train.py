"""Tests for model training."""

from src.train import train_model


def test_train_runs_on_small_dataset(tmp_path):
    """Training writes model artifacts for a small JSONL dataset."""
    data_path = tmp_path / "events.jsonl"
    data_path.write_text(
        "\n".join(
            '{"src_ip":"10.0.0.%d","dst_ip":"10.0.0.20",'
            '"src_port":500%d,"dst_port":443,"protocol":"tcp",'
            '"event_type":"network","failed_count":0,"bytes":%d}'
            % (index, index, 1000 + index)
            for index in range(1, 8)
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.pkl"
    result = train_model(data_path=data_path, model_path=model_path)
    assert model_path.exists()
    assert isinstance(result["threshold"], float)
