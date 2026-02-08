#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_label(raw: str) -> str:
    return raw.replace(".", "_").replace(":", "_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a manifest JSON for a cluster eval RUN_ID.")
    p.add_argument("--root", default="", help="Repo root (default: inferred from this script location)")
    p.add_argument("--run-id", required=True, help="RUN_ID prefix (matches results/structured/<run_id>_*)")
    p.add_argument("--hosts", default="", help="Comma-separated host list (optional)")
    p.add_argument("--labels", default="", help="Comma-separated labels (optional; must match host count)")
    p.add_argument(
        "--include-figures",
        action="store_true",
        help="Also include docs/figures/<run_id>_* files in the manifest.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    run_id = args.run_id

    struct_dir = root / "results" / "structured"
    raw_dir = root / "results" / "raw"
    fig_dir = root / "docs" / "figures"

    paths_set = set()
    if struct_dir.exists():
        for p in struct_dir.glob(f"{run_id}_*"):
            if p.is_file():
                paths_set.add(p)
    if raw_dir.exists():
        # Include both top-level files and files nested under run-specific raw dirs.
        for p in raw_dir.rglob(f"{run_id}_*"):
            if p.is_file():
                paths_set.add(p)
            elif p.is_dir():
                for fp in p.rglob("*"):
                    if fp.is_file():
                        paths_set.add(fp)
    if args.include_figures and fig_dir.exists():
        for p in fig_dir.glob(f"{run_id}_*"):
            if p.is_file():
                paths_set.add(p)

    paths: List[Path] = sorted(paths_set)
    files = [str(p.relative_to(root)) for p in paths]
    hashes = {str(p.relative_to(root)): _sha256(p) for p in paths}

    artifact_counts: Dict[str, int] = {}
    for p in paths:
        suffix = p.suffix.lstrip(".") or "no_ext"
        artifact_counts[suffix] = artifact_counts.get(suffix, 0) + 1

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if labels and len(labels) != len(hosts):
        raise SystemExit("--labels count must match --hosts count")

    nodes: List[Dict[str, Any]] = []
    for i, h in enumerate(hosts):
        label = labels[i] if labels else _sanitize_label(h)
        nodes.append({"label": label, "host": h})

    manifest: Dict[str, Any] = {
        "manifest_version": 1,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "nodes": nodes,
        "files": files,
        "summary": {
            "file_count": len(files),
            "artifact_counts": artifact_counts,
            "sha256": hashes,
        },
    }

    out_path = struct_dir / f"{run_id}_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
