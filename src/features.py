"""Feature extraction and persisted preprocessing artifacts."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import FEATURE_COLUMNS, MODELS_DIR, SENSITIVE_PORTS
from src.utils import is_internal_ip


def extract_features(record: dict) -> dict:
    """Convert one security event into domain features."""
    event_type = str(record.get("event_type", "other")).lower()
    failed_count = int(record.get("failed_count", 0) or 0)
    byte_count = float(record.get("bytes", 0) or 0)
    dst_port = int(record.get("dst_port", 0) or 0)
    return {
        "src_port": int(record.get("src_port", 0) or 0),
        "dst_port": dst_port,
        "failed_count": failed_count,
        "bytes": byte_count,
        "is_internal_src": is_internal_ip(str(record.get("src_ip", ""))),
        "is_internal_dst": is_internal_ip(str(record.get("dst_ip", ""))),
        "is_sensitive_port": int(dst_port in SENSITIVE_PORTS),
        "is_high_port": int(
            int(record.get("src_port", 0) or 0) > 1023 or dst_port > 1023
        ),
        "is_failed_auth": int(event_type == "auth" and failed_count > 0),
        "bytes_per_failed": byte_count / failed_count if failed_count else byte_count,
    }


def build_feature_matrix(
    records: list[dict],
    scaler: StandardScaler | None = None,
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, StandardScaler, dict[str, LabelEncoder]]:
    """Encode, scale, and persist the model feature matrix."""
    if not records:
        raise ValueError("records must contain at least one event")

    frame = pd.DataFrame([extract_features(record) for record in records])
    if fit:
        active_encoders = {
            "protocol": LabelEncoder().fit(
                [_category(record.get("protocol")) for record in records] + ["other"]
            ),
            "event_type": LabelEncoder().fit(
                [_category(record.get("event_type")) for record in records]
                + ["other"]
            ),
        }
        save_encoders(active_encoders)
    else:
        active_encoders = encoders or load_encoders()

    protocol_values = _encoded_categories(
        records, "protocol", active_encoders["protocol"]
    )
    event_type_values = _encoded_categories(
        records, "event_type", active_encoders["event_type"]
    )
    frame["protocol_encoded"] = active_encoders["protocol"].transform(
        protocol_values
    )
    frame["event_type_encoded"] = active_encoders["event_type"].transform(
        event_type_values
    )
    frame = frame.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    active_scaler = scaler or StandardScaler()
    matrix = (
        active_scaler.fit_transform(frame)
        if fit
        else active_scaler.transform(frame)
    )
    if fit:
        save_scaler(active_scaler)
    return matrix.astype(np.float64), active_scaler, active_encoders


def _category(value: object) -> str:
    """Normalize categorical values while keeping unknowns predictable."""
    return str(value or "other").strip().lower() or "other"


def _encoded_categories(
    records: list[dict], field: str, encoder: LabelEncoder
) -> list[str]:
    """Map categories absent from training to the shared ``other`` class."""
    known_categories = set(encoder.classes_)
    return [
        value if (value := _category(record.get(field))) in known_categories else "other"
        for record in records
    ]


def save_features(X: np.ndarray, path: Path) -> None:
    """Persist a feature matrix as a NumPy file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, X)


def load_features(path: Path) -> np.ndarray:
    """Load a persisted NumPy feature matrix."""
    return np.load(Path(path))


def save_scaler(scaler: StandardScaler, path: Path = MODELS_DIR / "scaler.pkl") -> None:
    """Persist the fitted scaler."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def save_encoders(encoders: dict[str, LabelEncoder], path: Path = MODELS_DIR) -> None:
    """Persist fitted protocol and event-type encoders."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders["protocol"], output_dir / "le_protocol.pkl")
    joblib.dump(encoders["event_type"], output_dir / "le_event_type.pkl")


def load_encoders(path: Path = MODELS_DIR) -> dict[str, LabelEncoder]:
    """Load fitted categorical encoders from the model directory."""
    model_dir = Path(path)
    return {
        "protocol": joblib.load(model_dir / "le_protocol.pkl"),
        "event_type": joblib.load(model_dir / "le_event_type.pkl"),
    }
