# API Reference

## `GET /health`

Returns `{"status":"ok"}`.

## `POST /predict`

Accepts the input contract from `schema.md` and returns `is_anomaly`, `anomaly_score`, `model`, and `threshold`.

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"src_ip":"10.0.0.10","dst_ip":"10.0.0.20","src_port":51000,"dst_port":22,"protocol":"tcp","event_type":"auth","failed_count":12,"bytes":5000}'
```
