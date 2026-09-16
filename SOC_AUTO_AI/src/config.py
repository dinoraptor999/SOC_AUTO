"""Application paths and model configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
PROCESSED_DATASET_DIR = DATASET_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
RAW_DATA_PATH = DATASET_DIR / "raw" / "sample.jsonl"
FEATURES_PATH = PROCESSED_DATASET_DIR / "features.npy"
MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
EVALUATION_LOG_PATH = OUTPUTS_DIR / "logs" / "evaluate.log"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"

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
    "failed_ratio",
    "is_high_failed",
    "is_scan_port",
    "is_night_time",
    "is_high_bytes",
    "is_low_bytes",
    "protocol_encoded",
    "event_type_encoded",
]
