"""Shared JSONL, logging, and IP helper functions."""

import json
import logging
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """Load non-empty JSON lines from a file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_json(data: object, path: Path) -> None:
    """Write JSON data, creating parent directories when needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=True)


def load_json(path: Path) -> object:
    """Read a JSON document."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def setup_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Create a consistently formatted application logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(handler.formatter)
            logger.addHandler(file_handler)
    return logger


def is_internal_ip(ip_str: str) -> int:
    """Return one for RFC1918/private addresses and zero otherwise."""
    import ipaddress

    try:
        return int(ipaddress.ip_address(ip_str).is_private)
    except ValueError:
        return 0
