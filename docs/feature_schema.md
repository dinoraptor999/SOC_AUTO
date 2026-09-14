# Feature Schema

| Feature | Type | Processing |
|---|---|---|
| `src_port`, `dst_port` | numeric | StandardScaler |
| `failed_count`, `bytes`, `bytes_per_failed` | numeric | StandardScaler |
| `is_internal_src`, `is_internal_dst` | binary | IP private-range encoding |
| `is_sensitive_port` | binary | Destination membership in the sensitive-port set |
| `is_high_port` | binary | Either port is above 1023 |
| `is_failed_auth` | binary | Auth event with at least one failure |
| `protocol_encoded` | categorical | Fitted `le_protocol.pkl`, then StandardScaler |
| `event_type_encoded` | categorical | Fitted `le_event_type.pkl`, then StandardScaler |

The output matrix has a stable order defined by `src.config.FEATURE_COLUMNS`.
