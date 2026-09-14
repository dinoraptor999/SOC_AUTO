# Architecture

The service is a small stateless FastAPI inference process backed by versioned model artifacts. Logstash normalizes Wazuh and Suricata events into the shared event schema. `features.py` derives network, protocol, event, and sensitivity signals. Training writes the Isolation Forest, scaler, and threshold to `models/`; the API reads them once during startup.

Elasticsearch remains the event search and retention layer. OPNsense can contribute firewall events through Logstash. TV4 consumes the prediction response to enrich incidents and trigger response workflows.
