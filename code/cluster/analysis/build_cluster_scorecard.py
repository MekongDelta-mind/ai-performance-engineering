#!/usr/bin/env python3
"""Build a unified cluster scorecard + bottleneck classification artifact."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _extract_label(path: Path, run_id: str, suffix: str) -> str | None:
    name = path.name
    prefix = f"{run_id}_"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    return name[len(prefix) : -len(suffix)]


def _detect_labels(structured_dir: Path, run_id: str) -> List[str]:
    suffixes = [
        "_gemm_gpu_sanity.csv",
        "_vllm_serve_sweep.csv",
        "_nvbandwidth.json",
        "_gpu_stream.json",
        "_fio.json",
    ]
    labels: set[str] = set()
    for suffix in suffixes:
        for p in structured_dir.glob(f"{run_id}_*{suffix}"):
            lbl = _extract_label(p, run_id, suffix)
            if lbl:
                labels.add(lbl)
    return sorted(labels)


def _peak_nccl(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {"peak_algbw_gbps": 0.0, "peak_busbw_gbps": 0.0}
    payload = _load_json(path)
    rows = payload.get("results") or []
    peak_algbw = max((_float(r.get("algbw_gbps")) for r in rows), default=0.0)
    peak_busbw = max((_float(r.get("busbw_gbps")) for r in rows), default=0.0)
    return {"peak_algbw_gbps": peak_algbw, "peak_busbw_gbps": peak_busbw}


def _label_metrics(structured_dir: Path, run_id: str, label: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"label": label}

    gemm_csv = structured_dir / f"{run_id}_{label}_gemm_gpu_sanity.csv"
    if gemm_csv.exists():
        rows = _read_csv_rows(gemm_csv)
        tflops = [_float(r.get("avg_tflops")) for r in rows if _float(r.get("avg_tflops")) > 0]
        out["gemm_max_tflops"] = max(tflops) if tflops else 0.0
        out["gemm_avg_tflops"] = (sum(tflops) / len(tflops)) if tflops else 0.0

    gpu_stream_json = structured_dir / f"{run_id}_{label}_gpu_stream.json"
    if gpu_stream_json.exists():
        payload = _load_json(gpu_stream_json)
        ops = {str(r.get("operation")): _float(r.get("bandwidth_gbps")) for r in (payload.get("operations") or [])}
        out["gpu_stream_peak_gbps"] = _float(payload.get("peak_bandwidth_gbps"))
        out["gpu_stream_triad_gbps"] = ops.get("triad", 0.0)
        out["gpu_stream_copy_gbps"] = ops.get("copy", 0.0)

    nvbw_json = structured_dir / f"{run_id}_{label}_nvbandwidth.json"
    if nvbw_json.exists():
        payload = _load_json(nvbw_json)
        key = payload.get("key_sum_gbps") or {}
        out["nvbandwidth_hbm_gbps"] = max(
            _float(key.get("device_to_device_memcpy_read_ce")),
            _float(key.get("device_to_device_memcpy_write_ce")),
        )
        out["nvbandwidth_pcie_h2d_gbps"] = _float(key.get("host_to_device_memcpy_ce"))
        out["nvbandwidth_pcie_d2h_gbps"] = _float(key.get("device_to_host_memcpy_ce"))
        out["nvbandwidth_peak_sum_gbps"] = _float(payload.get("peak_sum_gbps"))

    vllm_csv = structured_dir / f"{run_id}_{label}_vllm_serve_sweep.csv"
    if vllm_csv.exists():
        rows = _read_csv_rows(vllm_csv)
        parsed = []
        for row in rows:
            conc = int(_float(row.get("concurrency"), 0))
            parsed.append(
                {
                    "concurrency": conc,
                    "total_tok_s": _float(row.get("total_token_throughput")),
                    "p99_ttft_ms": _float(row.get("p99_ttft_ms")),
                    "p99_tpot_ms": _float(row.get("p99_tpot_ms")),
                }
            )
        parsed = [r for r in parsed if r["concurrency"] > 0 and r["total_tok_s"] > 0]
        parsed.sort(key=lambda r: r["concurrency"])
        if parsed:
            first = parsed[0]
            last = parsed[-1]
            out["vllm_min_concurrency"] = first["concurrency"]
            out["vllm_max_concurrency"] = last["concurrency"]
            out["vllm_tok_s_min_concurrency"] = first["total_tok_s"]
            out["vllm_tok_s_max_concurrency"] = last["total_tok_s"]
            out["vllm_throughput_gain_ratio"] = (
                (last["total_tok_s"] / first["total_tok_s"]) if first["total_tok_s"] > 0 else 0.0
            )
            out["vllm_p99_ttft_min_ms"] = first["p99_ttft_ms"]
            out["vllm_p99_ttft_max_ms"] = last["p99_ttft_ms"]
            out["vllm_p99_ttft_ratio"] = (
                (last["p99_ttft_ms"] / first["p99_ttft_ms"]) if first["p99_ttft_ms"] > 0 else 0.0
            )

    fio_json = structured_dir / f"{run_id}_{label}_fio.json"
    if fio_json.exists():
        payload = _load_json(fio_json).get("results") or {}
        seq_read = payload.get("seq_read") or {}
        seq_write = payload.get("seq_write") or {}
        out["fio_seq_read_mb_s"] = _float(seq_read.get("bw_mb_s"))
        out["fio_seq_write_mb_s"] = _float(seq_write.get("bw_mb_s"))

    return out


def _classify_bottleneck(summary: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    recommendations: List[str] = []
    bottleneck_type = "mixed"
    confidence = "low"

    comm_scale = _float(summary.get("nccl_multi_to_single_busbw_ratio"))
    vllm_ttft_ratio = _float(summary.get("vllm_p99_ttft_ratio"))
    vllm_gain = _float(summary.get("vllm_throughput_gain_ratio"))
    stream_hbm_ratio = _float(summary.get("gpu_stream_to_hbm_ratio"))
    hbm_gbps = _float(summary.get("nvbandwidth_hbm_gbps"))
    gemm_tflops = _float(summary.get("gemm_max_tflops"))
    pcie_h2d = _float(summary.get("nvbandwidth_pcie_h2d_gbps"))

    if comm_scale > 0 and comm_scale < 0.65:
        bottleneck_type = "communication-bound"
        confidence = "high"
        reasons.append(
            f"Multi-node NCCL bus bandwidth ratio is low ({comm_scale:.2f} vs single-node), indicating interconnect pressure."
        )
        recommendations.extend(
            [
                "Tune NCCL environment (CROSS_NIC, QPs, CTAs) and verify topology affinity.",
                "Profile collective overlap with compute using Nsight Systems NVTX ranges.",
            ]
        )
    elif vllm_ttft_ratio >= 3.0 and vllm_gain <= 2.2:
        bottleneck_type = "host-bound"
        confidence = "high"
        reasons.append(
            f"vLLM tail latency knee is severe (p99 TTFT ratio={vllm_ttft_ratio:.2f}) while throughput gain is limited ({vllm_gain:.2f}x)."
        )
        recommendations.extend(
            [
                "Inspect scheduler/data-path overhead and CPU-side batching limits.",
                "Run Nsight Systems + PyTorch trace to attribute host gaps and launch overhead.",
            ]
        )
    elif stream_hbm_ratio > 0 and stream_hbm_ratio < 0.55:
        bottleneck_type = "memory-bound"
        confidence = "medium"
        reasons.append(
            f"STREAM-like triad bandwidth is low relative to nvbandwidth HBM copy ({stream_hbm_ratio:.2f} ratio)."
        )
        recommendations.extend(
            [
                "Focus on memory access efficiency (coalescing, cache reuse, fusion, vectorization).",
                "Use Nsight Compute to inspect DRAM throughput and memory stalls on hot kernels.",
            ]
        )
    elif pcie_h2d > 0 and pcie_h2d < 20.0:
        bottleneck_type = "host-transfer-bound"
        confidence = "medium"
        reasons.append(f"Host-to-device transfer bandwidth is low ({pcie_h2d:.1f} GB/s).")
        recommendations.extend(
            [
                "Check PCIe generation/link width and NUMA placement.",
                "Reduce H2D pressure using pinned memory, prefetching, and overlap.",
            ]
        )
    elif gemm_tflops > 0 and hbm_gbps > 0 and gemm_tflops < 250.0 and hbm_gbps > 1500.0:
        bottleneck_type = "compute-bound"
        confidence = "medium"
        reasons.append(
            f"GEMM throughput is comparatively low ({gemm_tflops:.1f} TFLOPS) while memory bandwidth is healthy ({hbm_gbps:.1f} GB/s)."
        )
        recommendations.extend(
            [
                "Inspect occupancy, tensor-core utilization, and launch configuration.",
                "Use Nsight Compute to optimize math pipeline utilization.",
            ]
        )
    else:
        reasons.append("No single dominant subsystem bottleneck exceeded heuristic trigger thresholds.")
        recommendations.extend(
            [
                "Treat this run as mixed-bound; prioritize top workload hotspots.",
                "Run targeted deep-dive profiling on the slowest end-to-end workloads.",
            ]
        )

    return {
        "bottleneck_type": bottleneck_type,
        "confidence": confidence,
        "reasons": reasons,
        "recommendations": recommendations,
    }


def _fmt(v: Any, prec: int = 2) -> str:
    f = _float(v, default=float("nan"))
    if f != f:  # NaN
        return ""
    return f"{f:.{prec}f}"


def _build_markdown(payload: Dict[str, Any]) -> str:
    run_id = payload["run_id"]
    summary = payload["summary"]
    bottleneck = payload["bottleneck"]
    lines: List[str] = []
    lines.append(f"# Cluster Scorecard: `{run_id}`")
    lines.append("")
    lines.append(f"Generated: `{payload['generated_at_utc']}`")
    lines.append("")
    lines.append("## Unified KPIs")
    lines.append("")
    lines.append("| Domain | KPI | Value |")
    lines.append("|---|---|---:|")
    lines.append(f"| Compute | GEMM max TFLOPS | `{_fmt(summary.get('gemm_max_tflops'), 1)}` |")
    lines.append(f"| Memory | nvbandwidth HBM GB/s | `{_fmt(summary.get('nvbandwidth_hbm_gbps'), 1)}` |")
    lines.append(f"| Memory | STREAM-like triad GB/s | `{_fmt(summary.get('gpu_stream_triad_gbps'), 1)}` |")
    lines.append(f"| Communication | NCCL single-node peak busbw GB/s | `{_fmt(summary.get('nccl_single_peak_busbw_gbps'), 1)}` |")
    lines.append(f"| Communication | NCCL multi-node peak busbw GB/s | `{_fmt(summary.get('nccl_multi_peak_busbw_gbps'), 1)}` |")
    lines.append(f"| Communication | Multi/single busbw ratio | `{_fmt(summary.get('nccl_multi_to_single_busbw_ratio'), 2)}` |")
    lines.append(f"| Host transfer | nvbandwidth H2D GB/s | `{_fmt(summary.get('nvbandwidth_pcie_h2d_gbps'), 1)}` |")
    lines.append(f"| Workload | vLLM throughput gain ratio | `{_fmt(summary.get('vllm_throughput_gain_ratio'), 2)}` |")
    lines.append(f"| Workload | vLLM p99 TTFT ratio | `{_fmt(summary.get('vllm_p99_ttft_ratio'), 2)}` |")
    lines.append("")
    lines.append("## Bottleneck Classification")
    lines.append("")
    lines.append("| Classifier | Value |")
    lines.append("|---|---|")
    lines.append(f"| Dominant bottleneck | `{bottleneck['bottleneck_type']}` |")
    lines.append(f"| Confidence | `{bottleneck['confidence']}` |")
    lines.append("")
    lines.append("| Evidence |")
    lines.append("|---|")
    for reason in bottleneck.get("reasons", []):
        lines.append(f"| {reason} |")
    lines.append("")
    lines.append("| Recommended next actions |")
    lines.append("|---|")
    for rec in bottleneck.get("recommendations", []):
        lines.append(f"| {rec} |")
    lines.append("")
    lines.append("## Per-Node Metrics")
    lines.append("")
    lines.append(
        "| Label | GEMM max TFLOPS | nvbandwidth HBM GB/s | STREAM triad GB/s | vLLM tok/s gain | vLLM p99 TTFT ratio | fio seq read MB/s | fio seq write MB/s |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload.get("per_label_metrics", []):
        lines.append(
            f"| `{row.get('label','')}` | `{_fmt(row.get('gemm_max_tflops'), 1)}` | `{_fmt(row.get('nvbandwidth_hbm_gbps'), 1)}` | `{_fmt(row.get('gpu_stream_triad_gbps'), 1)}` | `{_fmt(row.get('vllm_throughput_gain_ratio'), 2)}` | `{_fmt(row.get('vllm_p99_ttft_ratio'), 2)}` | `{_fmt(row.get('fio_seq_read_mb_s'), 1)}` | `{_fmt(row.get('fio_seq_write_mb_s'), 1)}` |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cluster scorecard and bottleneck classification.")
    p.add_argument("--run-id", required=True, help="Run ID prefix")
    p.add_argument(
        "--structured-dir",
        default=str(Path(__file__).resolve().parents[1] / "results" / "structured"),
        help="Structured artifact directory",
    )
    p.add_argument("--output-json", default="", help="Output JSON path (default: <structured>/<run_id>_cluster_scorecard.json)")
    p.add_argument("--output-md", default="", help="Output markdown path (default: <structured>/<run_id>_cluster_scorecard.md)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    structured_dir = Path(args.structured_dir)
    run_id = args.run_id
    output_json = Path(args.output_json) if args.output_json else structured_dir / f"{run_id}_cluster_scorecard.json"
    output_md = Path(args.output_md) if args.output_md else structured_dir / f"{run_id}_cluster_scorecard.md"

    labels = _detect_labels(structured_dir, run_id)
    per_label = [_label_metrics(structured_dir, run_id, lbl) for lbl in labels]

    single_nccl = _peak_nccl(structured_dir / f"{run_id}_node1_nccl.json")
    multi_nccl = _peak_nccl(structured_dir / f"{run_id}_2nodes_nccl.json")
    nccl_ratio = 0.0
    if single_nccl["peak_busbw_gbps"] > 0:
        nccl_ratio = multi_nccl["peak_busbw_gbps"] / single_nccl["peak_busbw_gbps"]

    def _max_metric(key: str) -> float:
        return max((_float(row.get(key)) for row in per_label), default=0.0)

    # Prefer the first label for workload ratios (single-node vLLM sweep) when present.
    primary_row = per_label[0] if per_label else {}
    summary = {
        "gemm_max_tflops": _max_metric("gemm_max_tflops"),
        "nvbandwidth_hbm_gbps": _max_metric("nvbandwidth_hbm_gbps"),
        "nvbandwidth_pcie_h2d_gbps": _max_metric("nvbandwidth_pcie_h2d_gbps"),
        "nvbandwidth_pcie_d2h_gbps": _max_metric("nvbandwidth_pcie_d2h_gbps"),
        "gpu_stream_peak_gbps": _max_metric("gpu_stream_peak_gbps"),
        "gpu_stream_triad_gbps": _max_metric("gpu_stream_triad_gbps"),
        "nccl_single_peak_algbw_gbps": single_nccl["peak_algbw_gbps"],
        "nccl_single_peak_busbw_gbps": single_nccl["peak_busbw_gbps"],
        "nccl_multi_peak_algbw_gbps": multi_nccl["peak_algbw_gbps"],
        "nccl_multi_peak_busbw_gbps": multi_nccl["peak_busbw_gbps"],
        "nccl_multi_to_single_busbw_ratio": nccl_ratio if nccl_ratio > 0 else 0.0,
        "vllm_throughput_gain_ratio": _float(primary_row.get("vllm_throughput_gain_ratio")),
        "vllm_p99_ttft_ratio": _float(primary_row.get("vllm_p99_ttft_ratio")),
    }
    if summary["nvbandwidth_hbm_gbps"] > 0:
        summary["gpu_stream_to_hbm_ratio"] = summary["gpu_stream_triad_gbps"] / summary["nvbandwidth_hbm_gbps"]
    else:
        summary["gpu_stream_to_hbm_ratio"] = 0.0

    bottleneck = _classify_bottleneck(summary)
    payload = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels": labels,
        "per_label_metrics": per_label,
        "summary": summary,
        "bottleneck": bottleneck,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_build_markdown(payload), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
