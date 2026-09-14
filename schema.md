# Event Schema

## Input

| Field | Type | Description |
|---|---|---|
| `src_ip` | string | Source IPv4/IPv6 address |
| `dst_ip` | string | Destination address |
| `src_port` | integer | Source port |
| `dst_port` | integer | Destination port |
| `protocol` | string | `tcp`, `udp`, or `icmp` |
| `event_type` | string | `auth`, `network`, or `malware` |
| `failed_count` | integer | Failed attempts |
| `bytes` | number | Bytes transferred |

## Output

```json
{"is_anomaly": 0, "anomaly_score": 0.12, "model": "IsolationForest", "threshold": -0.08}
```

`is_anomaly` is 1 when the model score is below the learned threshold.
