# Integration Guide

TV2 (Logstash) should parse Wazuh `alerts.json`, Suricata `eve.json`, and OPNsense firewall records into the common fields in `schema.md`, then call `POST /predict`. Preserve the original event in Elasticsearch and store the response as enrichment fields.

TV4 can call the endpoint synchronously for triage or consume an asynchronous adapter built around the same contract. Retry transient HTTP failures, treat `503` as a missing model artifact, and keep the returned threshold and model name with the incident for auditability.
