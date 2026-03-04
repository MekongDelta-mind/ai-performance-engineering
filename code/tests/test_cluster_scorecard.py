from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_cluster_scorecard_detects_memory_bound(tmp_path: Path) -> None:
    run_id = "2026-03-03_scorecard_test"
    label = "node1"
    structured = tmp_path / "structured"
    structured.mkdir(parents=True, exist_ok=True)

    # GEMM sanity baseline.
    (structured / f"{run_id}_{label}_gemm_gpu_sanity.csv").write_text(
        "label,m,n,k,dtype,iters,avg_ms,p50_ms,p99_ms,avg_tflops,p50_tflops,p99_tflops\n"
        f"{label}_gpu0,16384,16384,16384,bf16,50,10.0,10.0,11.0,650.0,640.0,600.0\n",
        encoding="utf-8",
    )

    # nvbandwidth indicates healthy copy bandwidth.
    _write_json(
        structured / f"{run_id}_{label}_nvbandwidth.json",
        {
            "status": "ok",
            "key_sum_gbps": {
                "device_to_device_memcpy_read_ce": 3000.0,
                "device_to_device_memcpy_write_ce": 2900.0,
                "host_to_device_memcpy_ce": 28.0,
                "device_to_host_memcpy_ce": 27.0,
            },
        },
    )

    # STREAM-like triad is materially below copy bandwidth -> memory inefficiency.
    _write_json(
        structured / f"{run_id}_{label}_gpu_stream.json",
        {
            "status": "ok",
            "peak_bandwidth_gbps": 1200.0,
            "operations": [
                {"operation": "copy", "bandwidth_gbps": 1200.0},
                {"operation": "scale", "bandwidth_gbps": 1100.0},
                {"operation": "add", "bandwidth_gbps": 1000.0},
                {"operation": "triad", "bandwidth_gbps": 900.0},
            ],
        },
    )

    # Healthy NCCL scaling to avoid triggering communication-bound logic.
    _write_json(
        structured / f"{run_id}_node1_nccl.json",
        {"results": [{"algbw_gbps": 500.0, "busbw_gbps": 480.0}]},
    )
    _write_json(
        structured / f"{run_id}_2nodes_nccl.json",
        {"results": [{"algbw_gbps": 410.0, "busbw_gbps": 400.0}]},
    )

    # Workload sweep without strong host-knee symptoms.
    (structured / f"{run_id}_{label}_vllm_serve_sweep.csv").write_text(
        "model,tp,isl,osl,concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,gpu_util_mean_pct,gpu_util_p95_pct,mem_used_mean_mb,mem_used_max_mb,completed,failed\n"
        "test,1,128,64,1,10,1,100,200,10,10,12,5,5,6,50,60,1000,1200,10,0\n"
        "test,1,128,64,8,80,1,650,1200,14,13,18,8,7,9,70,80,1100,1300,80,0\n",
        encoding="utf-8",
    )

    out_json = structured / f"{run_id}_cluster_scorecard.json"
    out_md = structured / f"{run_id}_cluster_scorecard.md"
    cmd = [
        sys.executable,
        "cluster/analysis/build_cluster_scorecard.py",
        "--run-id",
        run_id,
        "--structured-dir",
        str(structured),
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_json.exists()
    assert out_md.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["bottleneck"]["bottleneck_type"] == "memory-bound"
    assert payload["summary"]["gpu_stream_to_hbm_ratio"] < 0.55
