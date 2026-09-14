# SOC Auto - AI/ML Anomaly Detection Service

AI/ML service for TV3 in a SOC Automation system. It normalizes Wazuh and Suricata events, extracts numeric features, and classifies outliers with Isolation Forest. The service is designed to integrate with Logstash, Elasticsearch, OPNsense, and the other SOC components.

## Pipeline

```text
Wazuh / Suricata -> Logstash -> JSON event -> features.py -> Isolation Forest -> FastAPI -> TV4
                                      \-> Elasticsearch (storage/search)
```

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/generate_sample_data.py
python -m src.train
python -m src.evaluate
pytest tests/ -v
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Then query `GET /health` or post an event to `POST /predict`. Docker users can run `docker compose up --build` after training artifacts exist in `models/`.

See [schema.md](schema.md) and the [docs](docs/) directory for the event contract, architecture, API reference, feature schema, and integration guidance.
