"""Application paths and model configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
RAW_DATA_PATH = DATASET_DIR / "raw" / "sample.jsonl"
MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"

SENSITIVE_PORTS = {22, 3389, 445, 1433, 3306}
DEFAULT_THRESHOLD = 0.65
FEATURE_COLUMNS = [
    "src_port",
    "dst_port",
    "failed_count",
    "bytes",
    "is_internal_src",
    "is_internal_dst",
    "is_sensitive_port",
    "is_high_port",
    "is_failed_auth",
    "bytes_per_failed",
    "protocol_encoded",
    "event_type_encoded",
]
