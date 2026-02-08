# Cluster Evaluation Harness

This folder holds a reproducible, strict cluster evaluation harness (discovery + benchmarks + plots) and a field report write-up.

Key rule (GPU benchmarks):
- GPU clock locking is **mandatory**. GPU benchmark scripts fail if clock locking cannot be acquired via the repo harness (`lock_gpu_clocks`).
- Practically, this usually means you must configure passwordless sudo for `nvidia-smi` clock locking (so `sudo -n true` succeeds).

## Quick Start

### 0) Run The Full Suite (Recommended)
This runs discovery + NCCL (1 node + 2 nodes) + vLLM serving sweep + GEMM sanity + fio + plots + manifest refresh:
```bash
scripts/run_cluster_eval_suite.sh \
  --run-id 2026-02-07_neocloud_case_study \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --oob-if <iface> \
  --socket-ifname <iface> \
  --nccl-ib-hca mlx5_0,mlx5_1,mlx5_4,mlx5_5 \
  --health-suite collectives \
  --model openai/gpt-oss-120b \
  --isl 1024 \
  --osl 1024 \
  --concurrency-range "32 64 128 256 512"
```
Optional GB200-focused diagnostics can be toggled in the same suite run:
```bash
  --fp4-suite-dir /path/to/cluster_perf_suite \
  --health-suite extended \
  --health-gdr --health-gdr-gpu 0 --health-gdr-mem-types 0,1 \
  --run-c2c --c2c-device 0 \
  --run-numa-mem-bw --numa-bytes 1073741824 --numa-iters 10 \
  --run-train-step --train-step-single-node --train-step-multi-node \
  --run-checkpoint-io --checkpoint-test-dir /tmp --checkpoint-bytes 4G
```
FP4 checks are enabled by default in `run_cluster_eval_suite.sh`. To skip them, pass `--disable-fp4`.
Optional high-impact cross-reference diagnostics:
```bash
  --enable-mamf --mamf-mode quick --mamf-concurrent \
  --enable-allreduce-stability --allreduce-payload-gib 2.0 --allreduce-iters 200 \
  --enable-allreduce-latency-comp --allreduce-latency-payload-gib 4.0 --allreduce-latency-chunks 1000 \
  --enable-allgather-control-plane --allgather-control-iters 2000 --allgather-control-warmup 200 \
  --enable-nccl-algo-comparison --nccl-algos Ring,Tree,NVLS,auto
```

### 0) Identity Snapshot And Uniqueness
Capture identity state (machine-id, hostname, SSH host keys) and log it for the field report:
```bash
scripts/setup.sh --label node1
```
Apply uniqueness fixes when needed (regenerate machine-id and SSH host keys, set hostname). Rotation is blocked unless explicitly overridden:
```bash
ALLOW_ID_ROTATION=1 ALLOW_SSH_KEY_ROTATION=1 scripts/setup.sh --label node2 --set-hostname node2 --regenerate-machine-id --regenerate-ssh-hostkeys --apply
```
Include peer ping checks in the readiness output:
```bash
scripts/setup.sh --label node1 --peers <peer_ip1,peer_ip2>
```
Append operator actions to a per-run JSONL log:
```bash
scripts/setup.sh --label node1 --log-ops
```
Outputs:
`results/structured/<run_id>_<label>_identity_pre.json`, `results/structured/<run_id>_<label>_identity_post.json`, `results/structured/<run_id>_<label>_readiness.json`, `results/raw/<run_id>_<label>_setup.log`, `results/raw/<run_id>_operator_actions.jsonl` (when `--log-ops` is used).

Validate operator log schema:
```bash
python3 scripts/validate_operator_log.py --input results/raw/<run_id>_operator_actions.jsonl
```

### 1) Discovery
```bash
scripts/collect_system_info.sh --output results/structured/<run_id>_meta.json --label node1
```
For all nodes (requires SSH access):
```bash
RUN_ID=2026-02-07 \
  scripts/run_discovery_all_nodes.sh \
    --hosts node1,node2 \
    --ssh-key ~/.ssh/ssh_key.pem
```

TCP sysctl snapshots (structured JSON for diffing):
```bash
RUN_ID=2026-02-07 \
  scripts/collect_tcp_sysctl_all_nodes.sh \
    --hosts node1,node2 \
    --ssh-key ~/.ssh/ssh_key.pem
```

One-shot: discovery + tcp sysctl + storage layout + manifest
```bash
RUN_ID=2026-02-07 \
  scripts/collect_discovery_and_tcp_sysctl.sh \
    --hosts node1,node2 \
    --ssh-key ~/.ssh/ssh_key.pem
```
This writes a manifest JSON to `results/structured/<run_id>_manifest.json` (includes `manifest_version`, file hashes, and artifact counts).
Schema: `docs/manifest_schema.md`.

After generating plots, refresh the manifest to include figures:
```bash
python3 scripts/write_manifest.py --run-id <run_id> --hosts node1,node2 --include-figures
```

### 1b) Enable Researcher Stack (Optional)
Dry-run first:
```bash
scripts/enable_researcher_stack.sh
```
Apply on a node:
```bash
scripts/enable_researcher_stack.sh --apply
```

### 1c) Cluster Health Suite (1 command)
Runs `iperf3` + `ib_write_bw` + `nccl-tests` + `torchrun` and writes raw logs under `results/raw/` and a single JSON summary under `results/structured/`:
```bash
scripts/run_cluster_health_suite.sh --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem
```
Run suite on a subset of GPUs (example: exclude GPU0 on each node):
```bash
scripts/run_cluster_health_suite.sh --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --gpus-per-node 3 --cuda-visible-devices 1,2,3
```
Extended run (also adds `ib_read_bw` + `ib_send_bw` + NCCL `alltoall_perf`):
```bash
scripts/run_cluster_health_suite.sh --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --extended
```
If you hit an NCCL NVLS failure like `transport/nvls.cc: NCCL WARN Cuda failure 801 'operation not supported'`, rerun with NVLS disabled:
```bash
scripts/run_cluster_health_suite.sh --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --extended --nccl-nvls-enable 0
```

Repeat the suite to quantify variance (base + extended per repetition):
```bash
scripts/run_cluster_health_suite_repeats.sh --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --repeats 3 --mode both --prefix 2026-02-07_suite_variance
```
Pass extra args through to the suite with `--` (example: NCCL-only repeats):
```bash
scripts/run_cluster_health_suite_repeats.sh --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --repeats 3 --mode base --prefix 2026-02-07_nccl_only -- --skip-iperf3 --skip-ib --skip-torchdist
```

Summarize variance across multiple suite summaries:
```bash
python3 analysis/summarize_cluster_health_suite_variance.py --glob 'results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu*_cluster_health_suite_summary.json' --output-md results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_variance.md --output-json results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_variance.json
```
Plot key metrics across repeats:
```bash
python3 analysis/plot_cluster_health_suite_variance.py --glob 'results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu*_cluster_health_suite_summary.json' --output docs/figures/2026-02-07_144800_cluster_health_suite_variance_ubuntu_metrics.png
```
Compare two suite summaries (flags regressions/improvements):
```bash
python3 analysis/compare_cluster_health_summaries.py --baseline results/structured/<baseline>_cluster_health_suite_summary.json --candidate results/structured/<candidate>_cluster_health_suite_summary.json --threshold 0.05 --output-md results/structured/<baseline>_vs_<candidate>.md --output-json results/structured/<baseline>_vs_<candidate>.json
```

### 2) Plotting (after results exist)
```bash
python3 analysis/plot_nccl.py --input results/structured/<run_id>_nccl.json --out-dir docs/figures --run-id <run_id>
python3 analysis/plot_vllm.py --input results/structured/<run_id>_vllm.csv --out-dir docs/figures --run-id <run_id>
python3 analysis/plot_vllm_serve_sweep.py --input results/structured/<run_id>_<label>_vllm_serve_sweep.csv --out-dir docs/figures --run-id <run_id>_<label>
python3 analysis/plot_fio.py --input results/structured/<run_id>_<label>_fio.json --out docs/figures/<run_id>_<label>_fio.png
```

### 2a) Benchmark A (Networking): NCCL `all_reduce_perf`
Single-node:
```bash
scripts/run_nccl_all_reduce.sh --run-id <run_id>_node1 --hosts localhost --label node1
python3 analysis/plot_nccl.py --input results/structured/<run_id>_node1_nccl.json --out-dir docs/figures --run-id <run_id>_node1
```

Multi-node (recommended explicit settings):
```bash
scripts/run_nccl_all_reduce.sh \
  --run-id <run_id>_2nodes \
  --hosts node1,node2 \
  --label node1node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --oob-if <iface> \
  --socket-ifname <iface> \
  --nccl-ib-hca mlx5_0,mlx5_1,mlx5_4,mlx5_5
python3 analysis/plot_nccl.py --input results/structured/<run_id>_2nodes_nccl.json --out-dir docs/figures --run-id <run_id>_2nodes
```

### 3) Storage (fio)
```bash
scripts/run_fio_bench.sh --run-id <run_id> --label <label> --test-dir <path>
python3 analysis/plot_fio.py --input results/structured/<run_id>_<label>_fio.json --out docs/figures/<run_id>_<label>_fio.png
```

### 4) Inference (vLLM online serving sweep)
```bash
scripts/repro/run_vllm_serve_sweep_container.sh \
  --run-id <run_id> \
  --label <label> \
  --model openai/gpt-oss-120b \
  --isl 1024 \
  --osl 1024 \
  --concurrency-range "32 64 128 256 512"
python3 analysis/plot_vllm_serve_sweep.py \
  --input results/structured/<run_id>_<label>_vllm_serve_sweep.csv \
  --out-dir docs/figures \
  --run-id <run_id>_<label>
```
This benchmark self-locks clocks (strict) and writes a clock-lock artifact to:
`results/structured/<run_id>_<label>_vllm_serve_sweep_clock_lock.json`.

### 5) Compute Sanity (GEMM per GPU, all nodes)
```bash
scripts/run_gemm_sanity_all_nodes.sh --run-id <run_id> --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem
python3 analysis/plot_gemm_bar.py --inputs results/structured/<run_id>_*_gemm_gpu_sanity.csv --output docs/figures/<run_id>_gemm_gpu_sanity.png --filter-m 16384
```

Optional: Long GEMM + 1 Hz telemetry (useful for chasing a few-% per-GPU or per-node deltas, or diagnosing power-cap behavior):
```bash
scripts/run_gemm_with_telemetry_all_nodes.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --gpus 0 \
  --iters 10000

python3 analysis/plot_gpu_telemetry.py \
  --csv results/raw/<run_id>_node1_gpu0_gemm_telemetry_query.csv --label node1_gpu0 \
  --csv results/raw/<run_id>_node2_gpu0_gemm_telemetry_query.csv --label node2_gpu0 \
  --out docs/figures/<run_id>_gpu0_telemetry.png \
  --title "GEMM Telemetry (GPU0): node1 vs node2"
```

### 5a) MAMF Finder (Maximum Achievable Matmul FLOPS)
Scans many matmul shapes to find the TRUE achievable TFLOPS ceiling for each GPU. This is the single most important compute diagnostic: it tells you the real performance bar (not theoretical peak), so you know when to stop optimizing.
```bash
scripts/run_mamf_finder_all_nodes.sh --run-id <run_id> --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --mode quick
scripts/run_mamf_finder_all_nodes.sh --run-id <run_id> --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --mode medium --concurrent
python3 analysis/plot_mamf.py --summary-inputs results/structured/<run_id>_*_mamf_summary.json --output docs/figures/<run_id>_mamf_straggler.png --mode straggler
```

### 5b) All-Reduce Stability Profiling (Network Jitter Detection)
Profiles a single large payload over many iterations to detect per-iteration bandwidth variance. A healthy network should show CV < 2%.
```bash
scripts/run_allreduce_stability.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --payload-gib 2.0 \
  --iters 200 \
  --socket-ifname <iface>
python3 analysis/plot_allreduce_stability.py --input results/structured/<run_id>_allreduce_stability.json --output docs/figures/<run_id>_allreduce_stability.png
```

### 5c) NCCL Algorithm Comparison (Ring vs Tree vs NVLS)
Tests NCCL algorithms explicitly to reveal if auto-selection is optimal:
```bash
scripts/run_nccl_algo_comparison.sh --run-id <run_id> --hosts node1,node2 --algos Ring,Tree,NVLS,auto --ssh-key ~/.ssh/ssh_key.pem --socket-ifname <iface>
python3 analysis/plot_nccl_algo_comparison.py --inputs results/structured/<run_id>_nccl_algo_*.json --output docs/figures/<run_id>_nccl_algo_comparison.png
```

### 5d) Concurrent GPU Straggler Detection
Run all GPUs simultaneously to find the straggler (slowest GPU sets training pace):
```bash
scripts/run_gemm_sanity_all_nodes.sh --run-id <run_id> --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --concurrent
```

### 5e) All-Reduce Latency Comparison (1x Large vs Many Small)
Compares one large all-reduce vs many smaller all-reduces with equivalent total payload, which highlights communication fragmentation overhead:
```bash
scripts/run_allreduce_latency_comp.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --payload-gib 4.0 \
  --chunks 1000 \
  --socket-ifname <iface>
python3 analysis/plot_allreduce_latency_comp.py --input results/structured/<run_id>_allreduce_latency_comp.json --output docs/figures/<run_id>_allreduce_latency_comp.png
```

### 5f) All-Gather Control-Plane Comparison
Quantifies the overhead of `all_gather_object` versus tensor collectives for control-path synchronization:
```bash
scripts/run_allgather_control_plane.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --iters 2000 \
  --warmup 200 \
  --socket-ifname <iface>
python3 analysis/plot_allgather_control_plane.py --input results/structured/<run_id>_allgather_control_plane.json --output docs/figures/<run_id>_allgather_control_plane.png
```

### 5g) GPUDirect RDMA Validation (IB Perftest + Latency)
Run BW + latency checks with perftest `--use_cuda` (and optional dmabuf) through the health suite:
```bash
scripts/run_cluster_health_suite.sh \
  --run-id <run_id>_health_gdr \
  --hosts node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --extended \
  --gdr \
  --gdr-gpu 0 \
  --gdr-mem-types 0,1 \
  --gdr-use-dmabuf
```
Structured output includes base IB + tagged `ib_gdr` entries in:
`results/structured/<run_id>_<label>_cluster_health_suite_summary.json`.

### 5h) Grace/GB200 C2C + NUMA Probes
CPU<->GPU memcpy benchmark (pageable/pinned/managed host memory):
```bash
scripts/run_c2c_memcpy_bench.sh --run-id <run_id> --label <label> --device 0
python3 analysis/plot_c2c_memcpy.py --input results/structured/<run_id>_<label>_c2c_memcpy.json --out-dir docs/figures --run-id <run_id>_<label>
```

NUMA memory bandwidth probe (CPU NUMA nodes + memory-only NUMA domains):
```bash
scripts/run_numa_mem_bw_all_nodes.sh --run-id <run_id> --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem
python3 analysis/plot_numa_mem_bw.py --input results/structured/<run_id>_<label>_numa_mem_bw.json --out docs/figures/<run_id>_<label>_numa_mem_bw.png
```

### 5i) End-To-End Train-Step Benchmark
Distributed tiny-transformer training step benchmark (forward+backward+optimizer), with app clocks captured per rank:
```bash
scripts/run_torchrun_transformer_train_step.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --gpus-per-node 4 \
  --oob-if <iface> \
  --nccl-ib-hca mlx5_0,mlx5_1,mlx5_4,mlx5_5 \
  --steps 30 --warmup-steps 5 --precision bf16 --fsdp 1
python3 analysis/plot_torchrun_train_step.py --input results/structured/<run_id>_<label>_torchrun_train_step.json --out docs/figures/<run_id>_<label>_torchrun_train_step.png
```

### 5j) Checkpoint I/O Benchmark
Checkpoint-like write/read throughput benchmark across nodes:
```bash
scripts/run_checkpoint_io_all_nodes.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --test-dir /tmp \
  --bytes 4G \
  --block-size 4M \
  --files 1 \
  --fsync 1
```
Outputs:
`results/structured/<run_id>_<label>_checkpoint_io.json` and
`results/structured/<run_id>_<label>_checkpoint_io.csv`.

### 5k) FP4 Coverage (DeepGEMM FP8xFP4)
Run FP4 smoke + grouped GEMM benchmark across all hosts:
```bash
scripts/run_fp4_checks_all_nodes.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --suite-dir /path/to/cluster_perf_suite \
  --preset auto \
  --warmup 5 \
  --iters 30
```
Outputs:
`results/structured/<run_id>_<label>_cluster_perf_fp4_platform.json`,
`results/structured/<run_id>_<label>_cluster_perf_fp4_smoke.json`,
`results/structured/<run_id>_<label>_cluster_perf_grouped_gemm_summary.json`,
`docs/figures/<run_id>_<label>_cluster_perf_grouped_gemm_tflops.png`.

### 6) Optional: Screenshot Repro Suite
Runs the commands/benchmarks shown in the case-study screenshots and writes raw logs under `results/raw/` (gitignored):
```bash
RUN_ID=2026-02-06_image_suite scripts/repro/run_image_suite.sh --run-id "$RUN_ID"
```

## Layout
```
cluster/
  analysis/               # plotting scripts
  docs/figures/           # generated plots
  env/requirements.txt    # plotting deps
  results/raw/            # raw logs
  results/structured/     # structured JSON/CSV
  scripts/                # discovery + run helpers
  field-report.md         # clean write-up (no results/raw links)
```

## Notes
- `results/raw/` is intentionally gitignored; the field report should link only to `results/structured/` and `docs/figures/`.
