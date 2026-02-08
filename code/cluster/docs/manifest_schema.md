# Manifest Schema

Current version: `1`

## Purpose
This document pins the manifest schema used by `scripts/collect_discovery_and_tcp_sysctl.sh` so future changes are explicit and traceable.

## Versioning
- Increment `manifest_version` when fields are added, removed, or meaningfully changed.
- Keep backwards compatibility notes in a new section for each version.

## Schema (v1)

| Field | Type | Description |
| --- | --- | --- |
| manifest_version | integer | Schema version. |
| run_id | string | Run identifier (e.g., `2026-02-05`). |
| timestamp_utc | string | ISO-8601 timestamp in UTC. |
| nodes | array | List of nodes in the run (`label` + `host`). |
| files | array | Relative paths of artifacts created for the run. |
| summary | object | Integrity summary and artifact counts. |
| summary.file_count | integer | Total number of files in `files`. |
| summary.artifact_counts | object | Counts by file extension (e.g., `json`, `txt`). |
| summary.sha256 | object | Mapping of relative file path to SHA-256 hash. |

## Example (v1)
```json
{
  "manifest_version": 1,
  "run_id": "2026-02-05",
  "timestamp_utc": "2026-02-05T04:55:12.345678+00:00",
  "nodes": [
    {"label": "node1", "host": "node1.example.internal"},
    {"label": "node2", "host": "node2.example.internal"}
  ],
  "files": [
    "results/structured/2026-02-05_node1_meta.json",
    "results/structured/2026-02-05_node1_tcp_sysctl.json"
  ],
  "summary": {
    "file_count": 2,
    "artifact_counts": {"json": 2},
    "sha256": {
      "results/structured/2026-02-05_node1_meta.json": "...",
      "results/structured/2026-02-05_node1_tcp_sysctl.json": "..."
    }
  }
}
```
