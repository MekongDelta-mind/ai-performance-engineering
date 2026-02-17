from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cluster_root() -> Path:
    return _repo_root() / "cluster"


def _run_cmd(cmd: List[str], *, cwd: Optional[Path] = None, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_ms": duration_ms,
        }
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "command": cmd,
            "returncode": None,
            "stdout": (exc.stdout or ""),
            "stderr": (exc.stderr or ""),
            "error": f"timeout after {timeout_seconds}s",
            "duration_ms": duration_ms,
        }
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "command": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
            "duration_ms": duration_ms,
        }


def _default_run_id_prefix() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def _default_primary_label() -> str:
    host = socket.gethostname() or "local"
    return host.split(".")[0] or "local"


def _smoke_eval(run_id: str, *, primary_label: Optional[str] = None, timeout_seconds: int = 120) -> Dict[str, Any]:
    cluster_root = _cluster_root()
    scripts_dir = cluster_root / "scripts"
    struct_dir = cluster_root / "results" / "structured"
    struct_dir.mkdir(parents=True, exist_ok=True)

    label = primary_label or _default_primary_label()
    meta_path = struct_dir / f"{run_id}_{label}_meta.json"

    collect_script = scripts_dir / "collect_system_info.sh"
    manifest_script = scripts_dir / "write_manifest.py"

    if not collect_script.exists():
        return {"success": False, "error": f"Missing script: {collect_script}"}
    if not manifest_script.exists():
        return {"success": False, "error": f"Missing script: {manifest_script}"}

    collect_cmd = ["bash", str(collect_script), "--output", str(meta_path), "--label", str(label)]
    collect = _run_cmd(collect_cmd, cwd=cluster_root, timeout_seconds=timeout_seconds)
    if collect.get("returncode") != 0:
        return {
            "success": False,
            "mode": "smoke",
            "run_id": run_id,
            "primary_label": label,
            "meta_path": str(meta_path),
            "collect": collect,
            "error": "collect_system_info failed",
        }

    manifest_cmd = ["python3", str(manifest_script), "--root", str(cluster_root), "--run-id", str(run_id)]
    manifest = _run_cmd(manifest_cmd, cwd=cluster_root, timeout_seconds=timeout_seconds)
    manifest_path = (manifest.get("stdout") or "").strip().splitlines()[-1] if manifest.get("stdout") else ""

    return {
        "success": bool(manifest.get("returncode") == 0),
        "mode": "smoke",
        "run_id": run_id,
        "primary_label": label,
        "meta_path": str(meta_path),
        "manifest_path": manifest_path or None,
        "collect": collect,
        "manifest": manifest,
    }


def run_cluster_eval_suite(
    *,
    mode: str = "smoke",
    run_id: Optional[str] = None,
    hosts: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    ssh_user: Optional[str] = None,
    ssh_key: Optional[str] = None,
    oob_if: Optional[str] = None,
    socket_ifname: Optional[str] = None,
    nccl_ib_hca: Optional[str] = None,
    primary_label: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the cluster eval suite (full) or a fast local smoke run.

    - mode=smoke:
      Writes `cluster/results/structured/<run_id>_<label>_meta.json` via collect_system_info.sh
      and refreshes `cluster/results/structured/<run_id>_manifest.json`.

    - mode=full:
      Invokes `cluster/scripts/run_cluster_eval_suite.sh` (requires --hosts).
    """
    run_id_value = run_id or _default_run_id_prefix()
    mode_value = (mode or "smoke").strip().lower()
    timeout = int(timeout_seconds) if timeout_seconds is not None else None

    if mode_value in {"smoke", "local", "mini"}:
        return _smoke_eval(run_id_value, primary_label=primary_label, timeout_seconds=timeout or 120)

    if mode_value not in {"full", "eval", "suite"}:
        return {"success": False, "error": f"Unknown mode: {mode!r} (expected smoke|full)"}

    cluster_root = _cluster_root()
    script = cluster_root / "scripts" / "run_cluster_eval_suite.sh"
    if not script.exists():
        return {"success": False, "error": f"Missing script: {script}"}

    hosts_list = hosts or []
    hosts_list = [h.strip() for h in hosts_list if isinstance(h, str) and h.strip()]
    if not hosts_list:
        return {"success": False, "error": "--hosts is required in full mode", "mode": "full", "run_id": run_id_value}

    cmd: List[str] = ["bash", str(script), "--hosts", ",".join(hosts_list), "--run-id", str(run_id_value)]
    if labels:
        labels_list = [l.strip() for l in labels if isinstance(l, str) and l.strip()]
        if labels_list:
            cmd.extend(["--labels", ",".join(labels_list)])
    if ssh_user:
        cmd.extend(["--ssh-user", str(ssh_user)])
    if ssh_key:
        cmd.extend(["--ssh-key", str(ssh_key)])
    if oob_if:
        cmd.extend(["--oob-if", str(oob_if)])
    if socket_ifname:
        cmd.extend(["--socket-ifname", str(socket_ifname)])
    if nccl_ib_hca:
        cmd.extend(["--nccl-ib-hca", str(nccl_ib_hca)])
    if primary_label:
        cmd.extend(["--primary-label", str(primary_label)])
    if extra_args:
        cmd.extend([str(x) for x in extra_args if str(x).strip()])

    result = _run_cmd(cmd, cwd=cluster_root, timeout_seconds=timeout)
    success = result.get("returncode") == 0
    return {"success": bool(success), "mode": "full", "run_id": run_id_value, "command": cmd, **result}


def validate_field_report_requirements(
    *,
    report: Optional[str] = None,
    notes: Optional[str] = None,
    template: Optional[str] = None,
    runbook: Optional[str] = None,
    canonical_run_id: Optional[str] = None,
    allow_run_id: Optional[List[str]] = None,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """Run the field-report validator script and return stdout/stderr."""
    root = _repo_root()
    script = root / "cluster" / "scripts" / "validate_field_report_requirements.sh"
    if not script.exists():
        return {"success": False, "error": f"Missing validator script: {script}"}

    cmd: List[str] = ["bash", str(script)]
    if report:
        cmd.extend(["--report", str(report)])
    if notes:
        cmd.extend(["--notes", str(notes)])
    if template:
        cmd.extend(["--template", str(template)])
    if runbook:
        cmd.extend(["--runbook", str(runbook)])
    if canonical_run_id:
        cmd.extend(["--canonical-run-id", str(canonical_run_id)])
    if allow_run_id:
        for rid in allow_run_id:
            if rid:
                cmd.extend(["--allow-run-id", str(rid)])

    result = _run_cmd(cmd, cwd=root, timeout_seconds=int(timeout_seconds))
    success = result.get("returncode") == 0
    payload: Dict[str, Any] = {"success": bool(success), **result}
    if not success and not payload.get("error"):
        rc = payload.get("returncode")
        payload["error"] = f"validator failed (returncode={rc})"
    return payload
