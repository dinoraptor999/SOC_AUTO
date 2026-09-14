"""Generate a labeled local dataset for development."""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dataset" / "raw" / "sample.jsonl"


def generate_records(count: int = 500) -> list[dict]:
    """Generate 70 percent normal and 30 percent anomalous records."""
    random.seed(42)
    records = []
    for index in range(count):
        abnormal = index >= int(count * 0.7)
        records.append({
            "src_ip": f"10.0.0.{random.randint(1, 254)}",
            "dst_ip": f"10.0.1.{random.randint(1, 254)}" if not abnormal else "8.8.8.8",
            "src_port": random.randint(40000, 60000),
            "dst_port": random.choice([22, 80, 443, 3389, 445]) if abnormal else random.choice([53, 80, 443]),
            "protocol": random.choice(["tcp", "udp"]),
            "event_type": random.choice(["auth", "malware"]) if abnormal else "network",
            "failed_count": random.randint(10, 50) if abnormal else random.randint(0, 2),
            "bytes": random.randint(300000, 1000000) if abnormal else random.randint(100, 5000),
            "label": int(abnormal),
        })
    return records


def main() -> None:
    """Write generated records as JSONL."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as file:
        for record in generate_records():
            file.write(json.dumps(record) + "\n")
    print(f"Wrote 500 records to {OUTPUT}")


if __name__ == "__main__":
    main()
