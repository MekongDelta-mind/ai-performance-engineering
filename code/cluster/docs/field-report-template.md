# Cluster Field Report Template

Last updated: YYYY-MM-DD

Rules:
- Do not link to `results/raw/` (raw is gitignored and for debugging only).
- Link only to `results/structured/` and `docs/figures/`.
- GPU benchmark runs are valid only if clock locking succeeded (include the clock-lock artifact in the run).

## TL;DR
- Hardware summary (nodes, GPUs/node, CPU, RAM, OS)
- 5-10 bullets: what is weird/new/interesting
- 2 bullets: what a small AI research team should care about

## Scope + Artifacts
- Nodes in-scope: <node list>
- Excluded nodes: <none|...>
- Primary RUN_ID: <YYYY-MM-DD_provider_story_...>

Manifest (file hashes + artifact counts):
- `results/structured/<RUN_ID>_manifest.json`

Discovery:
- `results/structured/<RUN_ID>_<label>_meta.json`
- `results/structured/<RUN_ID>_<label>_container_runtime.txt` (includes container runtime details + key CVE checks, e.g. CVE-2025-23266)

## Cluster Story (First Contact)
- Access / launcher UX (SSH, Slurm, K8s)
- Container story (Docker/Podman/Enroot)
- Egress constraints (model download reality)
- Observability (DCGM/IMEX/Fabric Manager)

## Weird / New / Interesting
1. <finding>
2. <finding>
3. <finding>

## Benchmark A (Networking Story): NCCL `all_reduce_perf`
- Why: explain intra-node vs inter-node scaling
- Config: GPUs used, message size range, how many nodes
- Repro commands:
  - `scripts/run_nccl_all_reduce.sh ...`
- Artifacts:
  - `results/structured/<RUN_ID>_nccl.json`
  - `docs/figures/<RUN_ID>_nccl_bw_vs_msg.png`
  - `docs/figures/<RUN_ID>_nccl_scaling_efficiency.png`
- Interpretation bullets: what does the curve say about the network/topology?

## Benchmark B (Inference Story): vLLM Online Serving
- Why: explain throughput vs concurrency and latency knees
- Config: model, TP, ISL/OSL, concurrency sweep
- Repro commands:
  - `scripts/repro/run_vllm_serve_sweep_container.sh ...`
  - `python3 analysis/plot_vllm_serve_sweep.py ...`
- Artifacts:
  - `results/structured/<RUN_ID>_<label>_vllm_serve_sweep.csv`
  - `results/structured/<RUN_ID>_<label>_vllm_serve_sweep.jsonl`
  - `results/structured/<RUN_ID>_<label>_vllm_serve_sweep_clock_lock.json`
  - `docs/figures/<RUN_ID>_<label>_vllm_serve_total_tok_s_vs_concurrency.png`
  - `docs/figures/<RUN_ID>_<label>_vllm_serve_ttft_vs_concurrency.png`
  - `docs/figures/<RUN_ID>_<label>_vllm_serve_tpot_vs_concurrency.png`

## Supporting: Compute Sanity (BF16 GEMM)
- Why: catch per-node/per-GPU deltas fast
- Repro:
  - `scripts/run_gemm_sanity_all_nodes.sh ...`
- Artifacts:
  - `results/structured/<RUN_ID>_<label>_gemm_gpu_sanity.csv`
  - `docs/figures/<RUN_ID>_<label>_gemm_gpu_sanity.png`

## Supporting: Storage (fio)
- Why: baseline sequential MB/s + random IOPS
- Repro:
  - `scripts/run_fio_bench.sh ...`
  - `python3 analysis/plot_fio.py ...`
- Artifacts:
  - `results/structured/<RUN_ID>_<label>_fio.json`
  - `docs/figures/<RUN_ID>_<label>_fio.png`

## Supporting: System Health Suite (Optional)
- `scripts/run_cluster_health_suite.sh ...`
- `analysis/plot_cluster_health_suite_variance.py ...`

## Supporting: GPUDirect RDMA (Optional, Recommended On GB200 Clusters)
- Why: validate host<->NIC<->GPU path and detect GDR-specific regressions.
- Repro:
  - `scripts/run_cluster_health_suite.sh --gdr --gdr-gpu 0 --gdr-mem-types 0,1 [--gdr-use-dmabuf] ...`
- Artifacts:
  - `results/structured/<RUN_ID>_<label>_cluster_health_suite_summary.json`
  - `ib_gdr` section in summary payload (tagged by `gdr_gpuX_memY[_dmabuf]`)

## Supporting: Grace/GB200 Memory Topology Probes (Optional)
- Why: characterize CPU<->GPU C2C behavior and memory-only NUMA domains.
- Repro:
  - `scripts/run_c2c_memcpy_bench.sh ...`
  - `scripts/run_numa_mem_bw_all_nodes.sh ...`
- Artifacts:
  - `results/structured/<RUN_ID>_<label>_c2c_memcpy.json`
  - `docs/figures/<RUN_ID>_<label>_c2c_memcpy_bw.png`
  - `docs/figures/<RUN_ID>_<label>_c2c_memcpy_lat.png`
  - `results/structured/<RUN_ID>_<label>_numa_mem_bw.json`
  - `docs/figures/<RUN_ID>_<label>_numa_mem_bw.png`

## Supporting: End-To-End Train Step (Optional)
- Why: confirm real train-step behavior (compute + comm + optimizer), not just collectives.
- Repro:
  - `scripts/run_torchrun_transformer_train_step.sh ...`
  - `python3 analysis/plot_torchrun_train_step.py ...`
- Artifacts:
  - `results/structured/<RUN_ID>_<label>_torchrun_train_step.json`
  - `docs/figures/<RUN_ID>_<label>_torchrun_train_step.png`

## Supporting: Checkpoint I/O (Optional)
- Why: baseline save/load throughput and fsync overhead for checkpoint workflows.
- Repro:
  - `scripts/run_checkpoint_io_all_nodes.sh ...`
- Artifacts:
  - `results/structured/<RUN_ID>_<label>_checkpoint_io.json`
  - `results/structured/<RUN_ID>_<label>_checkpoint_io.csv`

## Implications For Small AI Teams
- <bullet>
- <bullet>

## Appendix
- Discovery links
- Any tuning deltas (sysctl, MTU, NCCL env)
