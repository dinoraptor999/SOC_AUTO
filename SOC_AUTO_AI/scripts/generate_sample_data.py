"""Generate a reproducible labeled dataset for anomaly detection."""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dataset" / "raw" / "sample.jsonl"
NORMAL_COUNT = 350
ANOMALY_COUNT = 150


def generate_normal_record() -> dict:
    """Generate one normal network event."""
    return {
        "src_ip": f"10.0.40.{random.randint(1, 254)}",
        "dst_ip": f"10.0.30.{random.randint(1, 254)}",
        "src_port": random.randint(1024, 49151),
        "dst_port": random.choice([80, 443, 53, 8080]),
        "protocol": random.choice(["TCP", "UDP", "ICMP"]),
        "event_type": random.choice(["authentication_success", "connection_attempt"]),
        "failed_count": random.randint(0, 2),
        "bytes": random.randint(500, 50000),
        "is_anomaly_ground_truth": 0,
    }


def generate_anomaly_record() -> dict:
    """Generate one anomalous event with suspicious internal addressing."""
    return {
        "src_ip": f"10.0.40.{random.randint(1, 254)}",
        "dst_ip": f"10.0.30.{random.randint(1, 254)}",
        "src_port": random.randint(49152, 65535),
        "dst_port": random.choice([22, 23, 445, 3389, 1433]),
        "protocol": random.choice(["TCP", "UDP", "ICMP"]),
        "event_type": random.choice(["authentication_failed", "port_scan"]),
        "failed_count": random.randint(10, 100),
        "bytes": random.choice([
            random.randint(0, 100),
            random.randint(500000, 5000000),
        ]),
        "is_anomaly_ground_truth": 1,
    }


def generate_records() -> list[dict]:
    """Generate exactly 350 normal and 150 anomalous records."""
    random.seed(42)
    return [
        *(generate_normal_record() for _ in range(NORMAL_COUNT)),
        *(generate_anomaly_record() for _ in range(ANOMALY_COUNT)),
    ]


def main() -> None:
    """Write generated records as JSONL."""
    records = generate_records()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    print(f"Normal: {NORMAL_COUNT}")
    print(f"Anomaly: {ANOMALY_COUNT}")
    print(f"Total: {len(records)}")
    print(f"Wrote {len(records)} records to {OUTPUT}")
    print("Sample normal records:")
    for record in records[:3]:
        print(json.dumps(record))
    print("Sample anomaly records:")
    for record in records[NORMAL_COUNT:NORMAL_COUNT + 3]:
        print(json.dumps(record))


if __name__ == "__main__":
    main()
