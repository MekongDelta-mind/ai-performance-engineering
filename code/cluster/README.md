# Cluster Evaluation Harness

This folder contains a strict, reproducible cluster-evaluation harness (discovery + benchmarks + plots) and a field report workflow.

## Non-Negotiable Rules
- GPU benchmark runs are valid only when clock locking succeeds through the harness (`lock_gpu_clocks`).
- Preflight is mandatory for suite runs (`nvidia-persistenced`, `nvidia-imex`, `nvidia-dcgm`).
- Runtime/CVE evidence collection is enabled by default in discovery + health workflows (CVE-2025-23266 and CVE-2025-23267).
- Do not rotate machine identity / SSH host keys unless explicitly approved for a break-glass recovery.

## Quick Start

### 1) Portable baseline (recommended first run)
Use this when FP4 external dependencies are not guaranteed.

```bash
scripts/run_cluster_eval_suite.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --oob-if <iface> \
  --socket-ifname <iface> \
  --nccl-ib-hca mlx5_0,mlx5_1,mlx5_4,mlx5_5 \
  --health-suite extended \
  --disable-fp4
```

### 2) Full GB200 run (FP4 enabled)
Use this when suite/image dependencies are available.

```bash
scripts/run_cluster_eval_suite.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --oob-if <iface> \
  --socket-ifname <iface> \
  --nccl-ib-hca mlx5_0,mlx5_1,mlx5_4,mlx5_5 \
  --health-suite extended \
  --fp4-suite-dir /path/to/cluster_perf_suite \
  --fp4-image ghcr.io/jordannanos/cmax-compute:latest
```

### 2b) Full GB200 run (all extended checks enabled)
Use this when you want the complete multi-node package used by the field report (network + inference + FP4 + train-step + C2C + stability/composition/control-plane arcs).

```bash
scripts/run_cluster_eval_suite.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --oob-if <iface> \
  --socket-ifname <iface> \
  --nccl-ib-hca mlx5_0,mlx5_1,mlx5_4,mlx5_5 \
  --health-suite extended \
  --health-gdr \
  --health-gdr-gpu 0 \
  --health-gdr-mem-types 0,1 \
  --health-gdr-use-dmabuf \
  --fp4-suite-dir /path/to/cluster_perf_suite \
  --fp4-image ghcr.io/jordannanos/cmax-compute:latest \
  --run-c2c \
  --run-numa-mem-bw \
  --run-train-step \
  --train-step-single-node \
  --train-step-multi-node \
  --run-checkpoint-io \
  --enable-mamf \
  --mamf-mode quick \
  --mamf-concurrent \
  --enable-allreduce-stability \
  --allreduce-payload-gib 2.0 \
  --allreduce-iters 200 \
  --allreduce-warmup 20 \
  --enable-allreduce-latency-comp \
  --allreduce-latency-payload-gib 4.0 \
  --allreduce-latency-chunks 1000 \
  --allreduce-latency-iters 5 \
  --allreduce-latency-warmup 1 \
  --enable-allgather-control-plane \
  --allgather-control-iters 2000 \
  --allgather-control-warmup 200 \
  --enable-nccl-algo-comparison \
  --nccl-algos Ring,Tree,NVLS,auto
```

Optional extension for 2-node vLLM serving (Ray + TP across nodes):

```bash
  --run-vllm-multinode \
  --vllm-multinode-concurrency 16 \
  --vllm-multinode-num-prompts 64 \
  --vllm-multinode-ray-port 6379
```

### 3) Generate / refresh report plots and manifest
`run_cluster_eval_suite.sh` already does this. If you need to refresh later:

```bash
python3 scripts/write_manifest.py --run-id <run_id> --hosts node1,node2 --include-figures
```

## FP4 Notes
- `--fp4-suite-dir` and bootstrap `--sync-suite-dir` are local filesystem paths to a Cluster Perf checkout, not container image references.
- Valid path forms for suite-dir flags:
  - suite root (`<suite>/standalone/compute`)
  - `<suite>/standalone`
  - `<suite>/standalone/compute`
  - a parent directory containing a suite root
- FP4 execution is containerized:
  - grouped GEMM: `scripts/run_cluster_perf_grouped_gemm.sh` (`docker run ... --image ...`)
  - DeepGEMM smoke: `scripts/run_cluster_perf_fp4_smoke.sh` (`docker run ... --image ...`)
- FP4 checks now enforce a balanced attestation by default:
  - semantic source check on `standalone/compute/gemm-bench/grouped_gemm_bench.py` (GB200 UE8M0 markers)
  - grouped GEMM behavioral validation (`--require-deepgemm`) so unsupported DeepGEMM paths fail fast
  - per-host provenance capture (suite path/git if available, container image ID/digest, driver/CUDA)
  - cross-host consistency report at `results/structured/<run_id>_fp4_attestation_consistency.json`
- Provenance capture is metadata-only. It does not mutate host state, lock versions, or pin files.
- The GB200 grouped-GEMM compatibility patch is tracked in this repo:
  - `code/cluster_perf_patches/deepgemm_gb200_grouped_gemm_ue8m0.patch`
  - full apply/verify steps are in `docs/advanced-runbook.md`.
- Health-suite GDR behavior:
  - if perftest CUDA mode (`--use_cuda`) is unavailable, GDR checks are automatically disabled with a recorded warning
  - if a GDR subtest fails at runtime (for example MR allocation errors), the suite logs warnings and continues with the rest of the run

## Required Structured Outputs
- `results/structured/<run_id>_manifest.json`
- `results/structured/<run_id>_suite_steps.json`
- `results/structured/<run_id>_preflight_services.json`
- `results/structured/<run_id>_<label>_meta.json`
- `results/structured/<run_id>_<label>_container_runtime.txt`
- `results/structured/<run_id>_health_suite_<mode>_<label>_cluster_health_suite_summary.json`
- `results/structured/<run_id>_node1_nccl.json`
- `results/structured/<run_id>_2nodes_nccl.json`
- `results/structured/<run_id>_<label>_vllm_serve_sweep.csv`
- `results/structured/<run_id>_<label>_vllm_serve_sweep.jsonl`
- `results/structured/<run_id>_<label>_vllm_serve_sweep_clock_lock.json`
- `results/structured/<run_id>_<leader_label>_vllm_multinode_serve.json` (when `--run-vllm-multinode`)
- `results/structured/<run_id>_<leader_label>_vllm_multinode_serve.csv` (when `--run-vllm-multinode`)
- `results/structured/<run_id>_<leader_label>_vllm_multinode_serve.jsonl` (when `--run-vllm-multinode`)
- `results/structured/<run_id>_<leader_label>_vllm_multinode_leader_clock_lock.json` (when `--run-vllm-multinode`)
- `results/structured/<run_id>_<worker_label>_vllm_multinode_worker_clock_lock.json` (when `--run-vllm-multinode`)
- `results/structured/<run_id>_<label>_gemm_gpu_sanity.csv`
- `results/structured/<run_id>_<label>_fio.json`
- `results/structured/<run_id>_node_parity_summary.json`
- `results/structured/<run_id>_<label>_nvlink_topology.json`
- `results/structured/<run_id>_fp4_attestation_consistency.json` (when FP4 checks are enabled)

## Documentation Map
- Advanced runbook and optional diagnostics: `docs/advanced-runbook.md`
- Field report template: `docs/field-report-template.md`
- Manifest schema: `docs/manifest_schema.md`
- Current report write-up: `field-report.md`
- Validated notes ledger (claim-to-evidence index): `field-report-notes.md`

## Reference Evidence Package
- Canonical validated run: `2026-02-09_gb200_fullflags_all_0117`
- Artifact manifest: `results/structured/2026-02-09_gb200_fullflags_all_0117_manifest.json`
- Sanitized cluster metadata: `results/structured/2026-02-09_gb200_fullflags_all_0117_cluster_meta.json`
- Networking arc (data + plots):
  - `results/structured/2026-02-09_gb200_fullflags_all_0117_health_suite_extended_node1node2_cluster_health_suite_summary.json`
  - `docs/figures/2026-02-09_gb200_fullflags_all_0117_2nodes_nccl_bw_vs_msg.png`
- Inference arc (data + plots):
  - `results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_serve_sweep.csv`
  - `docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_vllm_serve_total_tok_s_vs_concurrency.png`
  - `docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_vllm_serve_ttft_vs_concurrency.png`
- Multi-node vLLM path evidence (latest attempt, strict-lock, structured failure captured):
  - `results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.json`
  - `results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.csv`
- NVLink/NVSwitch topology artifacts:
  - `results/structured/2026-02-09_gb200_fullflags_all_0117_node1_nvlink_topology.json`
  - `results/structured/2026-02-09_gb200_fullflags_all_0117_node2_nvlink_topology.json`
  - `docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_nvlink_topology.png`
  - `docs/figures/2026-02-09_gb200_fullflags_all_0117_node2_nvlink_topology.png`
- Story dashboard + parity summary:
  - `docs/figures/2026-02-09_gb200_fullflags_all_0117_cluster_story_dashboard.png`
  - `results/structured/2026-02-09_gb200_fullflags_all_0117_node_parity_summary.json`

## Notes
- `results/raw/` is intentionally gitignored and for debugging only.
- Field reports should link to `results/structured/` and `docs/figures/` artifacts.
