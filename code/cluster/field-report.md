# Cluster Perf Field Report (GB200, 2 Nodes)

Last updated: 2026-02-09.

## TL;DR
- In scope hosts: `node1`, `node2` (2 nodes); 4x GB200 per node (8 total); excluded nodes: none.
- Baseline run for stakeholder conclusions: `2026-02-09_gb200_fullflags_all_0117`.
- Latest clean 2-node run is healthy and reproducible: NCCL all-reduce peak bus bandwidth `840.07 GB/s` (16 GiB), NVLS not degraded.
- vLLM serving shows a clear latency knee: output throughput rises to `26.88k tok/s` at concurrency `512`, but mean TTFT grows to `~5479 ms`.
- Multi-node vLLM serving path is now first-class in the harness with strict lock evidence on both nodes; latest run failed before readiness with `ModuleNotFoundError: vllm.config.kernel`.
- NVLink/NVSwitch topology artifacts are now bundled for both nodes and show full intra-node `NV18` mesh connectivity.
- OOB TCP is only `~7.72/7.53 Gbps` (fwd/rev), so treat it as control/bootstrap, not data plane.
- One-off GPU degradations and GEMM-collapses were transient: Either required reset or self-cleared under immediate locked rerun.
- Intentionally hard-requiring DCGM during preflight and before/after state is recorded per host.
- Historical incident evidence is retained only where it changes recommendations.
- GB200-focused harness updates should be upstreamed into independent `<cluster_perf_suite>`.

## TL;DR Evidence Anchors
- Scope + baseline package: [results/structured/2026-02-09_gb200_fullflags_all_0117_manifest.json](results/structured/2026-02-09_gb200_fullflags_all_0117_manifest.json), [latest_cluster_meta][latest_cluster_meta]
- NCCL health + NVLS behavior: [results/structured/2026-02-09_gb200_fullflags_all_0117_health_suite_extended_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-09_gb200_fullflags_all_0117_health_suite_extended_node1node2_cluster_health_suite_summary.json), [docs/figures/2026-02-09_gb200_fullflags_all_0117_2nodes_nccl_bw_vs_msg.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_2nodes_nccl_bw_vs_msg.png), [docs/figures/2026-02-09_gb200_fullflags_all_0117_nccl_algo_comparison.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_nccl_algo_comparison.png)
- Inference latency knee: [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_serve_sweep.csv](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_serve_sweep.csv), [tok_s_vs_conc][tok_s_vs_conc], [ttft_vs_conc][ttft_vs_conc]
- Multinode vLLM path status: [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.json](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.json), [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.csv](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.csv)
- NVLink/NVSwitch topology artifacts: [node1_nvlink_topology_json][node1_nvlink_topology_json], [node2_nvlink_topology_json][node2_nvlink_topology_json], [node1_nvlink_topology_png][node1_nvlink_topology_png], [node2_nvlink_topology_png][node2_nvlink_topology_png]
- OOB vs IB gap: [results/structured/2026-02-09_gb200_fullflags_all_0117_iperf3_oob_tcp.json](results/structured/2026-02-09_gb200_fullflags_all_0117_iperf3_oob_tcp.json), [docs/figures/2026-02-09_gb200_fullflags_all_0117_iperf3_oob_tcp.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_iperf3_oob_tcp.png), [docs/figures/2026-02-09_gb200_fullflags_all_0117_2nodes_nccl_bw_vs_msg.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_2nodes_nccl_bw_vs_msg.png)
- Transient anomaly evidence (kept as incident context): [results/structured/2026-02-08_ssh_key_full_suite_r2_node2_gemm_gpu_sanity.csv](results/structured/2026-02-08_ssh_key_full_suite_r2_node2_gemm_gpu_sanity.csv), [results/structured/2026-02-08_node2_gpu2_diag_pre_reset_node2_gemm_gpu_sanity.csv](results/structured/2026-02-08_node2_gpu2_diag_pre_reset_node2_gemm_gpu_sanity.csv), [docs/figures/2026-02-08_node2_gpu2_transient_gemm_tflops.png](docs/figures/2026-02-08_node2_gpu2_transient_gemm_tflops.png)
- Preflight/DCGM policy evidence: [results/structured/2026-02-09_gb200_fullflags_all_0117_preflight_services.json](results/structured/2026-02-09_gb200_fullflags_all_0117_preflight_services.json), [results/structured/2026-02-08_test_preflight_dcgm_before_after_node1node2_preflight_services.json](results/structured/2026-02-08_test_preflight_dcgm_before_after_node1node2_preflight_services.json), [operator_state_snapshot][operator_state_snapshot]

## Cluster Story (First Contact)
- First-contact timeline (UTC, from canonical run logs):
  - `01:16:36` bootstrap completed on both nodes.
  - `01:16:40` strict preflight completed (`persistenced`/`imex`/`dcgm` checks).
  - `01:17:22` first clean 2-node NCCL run completed.
  - `01:20:09` extended health suite finished.
  - `01:34:13` first full vLLM concurrency sweep completed.
  - `01:50:22` final manifest refresh completed.
- Time-to-first-multi-node-signal was short (`~1 minute` from preflight completion to first 2-node NCCL completion), but only because interface pinning and service policy were already codified in the harness.
- Cluster is HPC-flavored: strong IB/NCCL behavior with weak OOB TCP behavior.
- Largest first-contact friction points were operational, not kernel-level (service readiness + launch hygiene + queue discipline).
- Story evidence bundle: [latest_health][latest_health], [preflight_latest][preflight_latest], [docs/figures/2026-02-09_gb200_fullflags_all_0117_2nodes_nccl_bw_vs_msg.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_2nodes_nccl_bw_vs_msg.png), [docs/figures/2026-02-09_gb200_fullflags_all_0117_iperf3_oob_tcp.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_iperf3_oob_tcp.png), [cluster_story_dashboard][cluster_story_dashboard], [operator_state_snapshot][operator_state_snapshot]

## Normal vs Weird Log
| Area | Normal (clean baseline) | Weird (incident / edge case) | Evidence |
| --- | --- | --- | --- |
| NCCL multi-node | all-reduce peak `840.07 GB/s` with stable curve shape | historical low-band regime `~529.64 GB/s` | [latest_health][latest_health], [results/structured/2026-02-07_224500_nccl_16g_baseline_ppr4_bindnone_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-07_224500_nccl_16g_baseline_ppr4_bindnone_node1node2_cluster_health_suite_summary.json), [docs/figures/2026-02-07_nccl_allreduce_bimodal_overlay.png](docs/figures/2026-02-07_nccl_allreduce_bimodal_overlay.png) |
| Service state | preflight enforces `persistenced`/`imex`/`dcgm` healthy before run | missing service readiness broke NVLS init and vLLM startup | [preflight_latest][preflight_latest], [results/structured/2026-02-08_025442_cloud_eval_full_health_suite_extended_node1node2_nccl_all_reduce_perf.error_excerpt.txt](results/structured/2026-02-08_025442_cloud_eval_full_health_suite_extended_node1node2_nccl_all_reduce_perf.error_excerpt.txt), [results/structured/2026-02-08_025442_cloud_eval_full_node1_vllm_serve_sweep_sweep_log.txt](results/structured/2026-02-08_025442_cloud_eval_full_node1_vllm_serve_sweep_sweep_log.txt) |
| Inference serving | predictable throughput rise through `c=256` | strong TTFT knee at `c=512` (`~5479 ms`) | [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_serve_sweep.csv](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_serve_sweep.csv), [tok_s_vs_conc][tok_s_vs_conc], [ttft_vs_conc][ttft_vs_conc] |
| Inference serving (multi-node path) | new harness path executes with strict lock metadata on both hosts | current attempt fails during Ray worker import (`No module named 'vllm.config.kernel'`) before server readiness | [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.json](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.json), [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.csv](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.csv) |
| GEMM per-GPU | `node2_gpu2` in-family (`~1530.80 TFLOPS`) in clean baseline | one-off collapse (`~709 TFLOPS`) that recovered on immediate rerun (`~1548.7 TFLOPS`) | [results/structured/2026-02-09_gb200_fullflags_all_0117_node2_gemm_gpu_sanity.csv](results/structured/2026-02-09_gb200_fullflags_all_0117_node2_gemm_gpu_sanity.csv), [results/structured/2026-02-08_ssh_key_full_suite_r2_node2_gemm_gpu_sanity.csv](results/structured/2026-02-08_ssh_key_full_suite_r2_node2_gemm_gpu_sanity.csv), [results/structured/2026-02-08_node2_gpu2_diag_pre_reset_node2_gemm_gpu_sanity.csv](results/structured/2026-02-08_node2_gpu2_diag_pre_reset_node2_gemm_gpu_sanity.csv), [docs/figures/2026-02-08_node2_gpu2_transient_gemm_tflops.png](docs/figures/2026-02-08_node2_gpu2_transient_gemm_tflops.png) |

## Benchmark A (Networking Story)
- Current full-flags baseline (`2026-02-09_gb200_fullflags_all_0117`) demonstrates strong multi-node fabric behavior with stable collectives.
- Key metrics:
  - IB write bandwidth: `~387.14 Gbps` per active HCA.
  - NCCL max bus bandwidths: all-reduce `840.07 GB/s`, all-gather `655.39 GB/s`, reduce-scatter `675.43 GB/s`, alltoall `604.81 GB/s`.
  - torch distributed all-reduce sanity max: `715.64 GB/s`.
- Charts:
  - 2-node NCCL bus bandwidth: [docs/figures/2026-02-09_gb200_fullflags_all_0117_2nodes_nccl_bw_vs_msg.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_2nodes_nccl_bw_vs_msg.png)
  - 2-node NCCL scaling efficiency: [docs/figures/2026-02-09_gb200_fullflags_all_0117_2nodes_nccl_scaling_efficiency.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_2nodes_nccl_scaling_efficiency.png)

## Benchmark B (Inference Story)
- vLLM (`openai/gpt-oss-120b`, TP=4, ISL/OSL=1024/1024) shows throughput scaling with a clear latency knee.
- Key metrics:
  - `c=32`: output `6907.77 tok/s`, mean TTFT `184.53 ms`.
  - `c=256`: output `24876.11 tok/s`, mean TTFT `780.76 ms`.
  - `c=512`: output `26879.58 tok/s`, mean TTFT `5478.71 ms`.
- Charts:
  - tokens/sec vs concurrency: [tok_s_vs_conc][tok_s_vs_conc]
  - TTFT vs concurrency: [ttft_vs_conc][ttft_vs_conc]
  - TPOT vs concurrency: [docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_vllm_serve_tpot_vs_concurrency.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_vllm_serve_tpot_vs_concurrency.png)
- Multi-node serving path status (new harness path, TP=8 across `node1,node2`):
  - Run attempt: `2026-02-09_gb200_fullflags_all_0117` with `isl=512`, `osl=256`, `concurrency=16`, `num_prompts=64`.
  - Outcome: failed before readiness due Ray-worker import error `No module named 'vllm.config.kernel'` (`failure_reason` in structured summary).
  - Structured outputs: [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.json](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.json), [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.csv](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.csv)

## Node Parity Snapshot (node1 vs node2)
- Structured summary: [node_parity_summary][node_parity_summary]
- Dashboard plot (benchmark arcs + parity): [cluster_story_dashboard][cluster_story_dashboard]
- Current parity table (same workload, same clock-lock policy):

| Metric | node1 | node2 | node2/node1 | Notes |
| --- | ---: | ---: | ---: | --- |
| GEMM mean (BF16, avg TFLOPS across 4 GPUs) | `1537.67` | `1531.31` | `0.996x` | tightly matched |
| GEMM min (BF16, TFLOPS) | `1495.56` | `1499.96` | `1.003x` | no persistent straggler in clean run |
| NUMA local memcpy BW (GB/s, peak probed node) | `134.50` | `136.03` | `1.011x` | essentially parity |
| fio seq read (MB/s) | `706.93` | `n/a` | `n/a` | node2 fio not collected in this baseline |

## NVLink/NVSwitch Topology Snapshot
- Dedicated topology summaries are now bundled per node:
  - [node1_nvlink_topology_json][node1_nvlink_topology_json]
  - [node2_nvlink_topology_json][node2_nvlink_topology_json]
- Dedicated topology figures:
  - [node1_nvlink_topology_png][node1_nvlink_topology_png]
  - [node2_nvlink_topology_png][node2_nvlink_topology_png]
- Both nodes show a full 4-GPU `NV18` mesh (`6/6` GPU pairs on each node), which is consistent with the high intra-node collective behavior seen in Benchmark A.

## GB200-Focused Extensions (Enabled in this run)
- All-reduce stability (`2 GiB`, 200 iters): mean busbw `809.65 GB/s`, CV `1.687%`, jitter assessment `good`.
  - Data: [results/structured/2026-02-09_gb200_fullflags_all_0117_allreduce_stability.json](results/structured/2026-02-09_gb200_fullflags_all_0117_allreduce_stability.json)
  - Chart: [docs/figures/2026-02-09_gb200_fullflags_all_0117_allreduce_stability.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_allreduce_stability.png)
- All-reduce latency composition (`4 GiB` target): one large collective (`818.15 GB/s`) vs many small chunks (`122.21 GB/s`) gives a `6.69x` bandwidth ratio.
  - Data: [results/structured/2026-02-09_gb200_fullflags_all_0117_allreduce_latency_comp.json](results/structured/2026-02-09_gb200_fullflags_all_0117_allreduce_latency_comp.json)
  - Chart: [docs/figures/2026-02-09_gb200_fullflags_all_0117_allreduce_latency_comp.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_allreduce_latency_comp.png)
- Control-plane collective overhead: `all_reduce_tensor` is fastest (`0.1867 ms` mean) vs `all_gather_tensor` (`0.2984 ms`) and `all_gather_object` (`1.5705 ms`).
  - Data: [results/structured/2026-02-09_gb200_fullflags_all_0117_allgather_control_plane.json](results/structured/2026-02-09_gb200_fullflags_all_0117_allgather_control_plane.json)
  - Chart: [docs/figures/2026-02-09_gb200_fullflags_all_0117_allgather_control_plane.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_allgather_control_plane.png)
- NCCL algorithm comparison: NVLS (`839.33 GB/s`) and `auto` (`839.09 GB/s`) significantly outperform Ring (`698.28`) and Tree (`547.22`).
  - Data: [results/structured/2026-02-09_gb200_fullflags_all_0117_nccl_algo_comparison.json](results/structured/2026-02-09_gb200_fullflags_all_0117_nccl_algo_comparison.json)
  - Chart: [docs/figures/2026-02-09_gb200_fullflags_all_0117_nccl_algo_comparison.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_nccl_algo_comparison.png)
- Grace-Blackwell C2C memcpy path (`node1`): pinned transfer peaks at `124.88/124.52 Gbps` (H2D/D2H), with 4-byte pinned latency `~2.01/1.87 us`.
  - Data: [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_c2c_memcpy.json](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_c2c_memcpy.json)
  - Charts: [docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_c2c_memcpy_bw.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_c2c_memcpy_bw.png), [docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_c2c_memcpy_lat.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_c2c_memcpy_lat.png)
- Train-step sanity (BF16/FSDP): single-node `102,718 tok/s` vs multi-node `206,398 tok/s` with similar step time (`~0.159 s`).
  - Data: [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_single_node_torchrun_train_step.json](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_single_node_torchrun_train_step.json), [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_multinode_torchrun_train_step.json](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_multinode_torchrun_train_step.json)
  - Charts: [docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_single_node_torchrun_train_step.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_single_node_torchrun_train_step.png), [docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_multinode_torchrun_train_step.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_multinode_torchrun_train_step.png)
- FP4 smoke skew guard passed: max pairwise median DeepGEMM gap `0.96%` (`node1` vs `node2`), below `5.0%` threshold.
  - Data: [results/structured/2026-02-09_gb200_fullflags_all_0117_fp4_smoke_skew_guard.json](results/structured/2026-02-09_gb200_fullflags_all_0117_fp4_smoke_skew_guard.json)
  - Grouped GEMM summaries: [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_cluster_perf_grouped_gemm_summary.json](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_cluster_perf_grouped_gemm_summary.json), [results/structured/2026-02-09_gb200_fullflags_all_0117_node2_cluster_perf_grouped_gemm_summary.json](results/structured/2026-02-09_gb200_fullflags_all_0117_node2_cluster_perf_grouped_gemm_summary.json)
- MAMF straggler check (all 8 GPUs, concurrent quick mode): peak BF16 matmul spans `1568.04` to `1672.77 TFLOPS` (`~6.26%` spread).
  - Data: [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_gpu0_mamf_summary.json](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_gpu0_mamf_summary.json), [results/structured/2026-02-09_gb200_fullflags_all_0117_node2_gpu3_mamf_summary.json](results/structured/2026-02-09_gb200_fullflags_all_0117_node2_gpu3_mamf_summary.json)
  - Chart: [docs/figures/2026-02-09_gb200_fullflags_all_0117_mamf_straggler.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_mamf_straggler.png)

## Weird / New / Interesting Findings

### 1) WEIRD (historical, root-caused): NCCL low-band regime from stuck node1 physical GPU0 SM clock
- What happened: A historical run entered a low-band regime at `~529.64 GB/s` all-reduce peak vs normal `~840.55 GB/s`.
- Why it matters: This was a hardware-state anomaly, not workload variance.
- Data:
  - Low-band historical run: [results/structured/2026-02-07_224500_nccl_16g_baseline_ppr4_bindnone_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-07_224500_nccl_16g_baseline_ppr4_bindnone_node1node2_cluster_health_suite_summary.json)
  - High-band historical run: [results/structured/2026-02-07_140642_cluster_health_suite_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-07_140642_cluster_health_suite_node1node2_cluster_health_suite_summary.json)
- Visualization:
  - Regime overlay: [docs/figures/2026-02-07_nccl_allreduce_bimodal_overlay.png](docs/figures/2026-02-07_nccl_allreduce_bimodal_overlay.png)
- Action: Keep per-GPU clock telemetry + preflight + repeatability checks in standard validation.

### 2) WEIRD (historical incident, mitigated): service-health outage broke NCCL NVLS init and container startup
- What happened: Historical incident run failed NCCL NVLS init (`transport/nvls.cc`) and vLLM container startup (`/run/nvidia-persistenced/socket` missing).
- Why it matters: service state can invalidate both communication and serving.
- Data:
  - NCCL failure excerpt: [results/structured/2026-02-08_025442_cloud_eval_full_health_suite_extended_node1node2_nccl_all_reduce_perf.error_excerpt.txt](results/structured/2026-02-08_025442_cloud_eval_full_health_suite_extended_node1node2_nccl_all_reduce_perf.error_excerpt.txt)
  - vLLM container failure log: [results/structured/2026-02-08_025442_cloud_eval_full_node1_vllm_serve_sweep_sweep_log.txt](results/structured/2026-02-08_025442_cloud_eval_full_node1_vllm_serve_sweep_sweep_log.txt)
- Visualization:
  - NVLS on/off impact: [docs/figures/2026-02-08_nvls_on_off_allreduce_busbw.png](docs/figures/2026-02-08_nvls_on_off_allreduce_busbw.png)
  - Operator state snapshot: [operator_state_snapshot][operator_state_snapshot]
- Action: keep strict preflight mandatory before any benchmark and health-suite run.

### 3) NOTABLE: DCGM is now hard-required with before/after evidence
- What happened: In historical discovery, DCGM was asymmetric across nodes; preflight now hard-requires DCGM and records before/after/start-by-preflight.
- Why it matters: prevents “silent blind” monitoring runs.
- Data:
  - Historical before/after incident check: [results/structured/2026-02-08_test_preflight_dcgm_before_after_node1node2_preflight_services.json](results/structured/2026-02-08_test_preflight_dcgm_before_after_node1node2_preflight_services.json)
  - Latest clean baseline preflight: [preflight_latest][preflight_latest]
- Visualization:
  - Operator state snapshot: [operator_state_snapshot][operator_state_snapshot]
- Action: keep provider restart policy decision explicit; current unit policy remains `Restart=on-abort`.

### 4) NOTABLE: OOB TCP is much slower than IB and should remain bootstrap-only
- What happened: OOB TCP is `~7.72/7.53 Gbps` in clean baseline, while IB path sustains far higher collective throughput.
- Why it matters: interface/port pinning and control-plane assumptions determine launch reliability.
- Data:
  - OOB throughput: [results/structured/2026-02-09_gb200_fullflags_all_0117_iperf3_oob_tcp.json](results/structured/2026-02-09_gb200_fullflags_all_0117_iperf3_oob_tcp.json)
  - Health summary: [latest_health][latest_health]
- Visualization:
  - OOB TCP chart: [docs/figures/2026-02-09_gb200_fullflags_all_0117_iperf3_oob_tcp.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_iperf3_oob_tcp.png)
- Action: keep explicit OOB/socket/NCCL HCA pinning in all multi-node recipes.

### 5) NOTABLE: inference latency knee is strong and actionable
- What happened: throughput improves steadily, but TTFT rises sharply at high concurrency.
- Why it matters: user-facing SLOs require explicit concurrency caps, not throughput-only tuning.
- Data:
  - Sweep CSV: [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_serve_sweep.csv](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_serve_sweep.csv)
  - Sweep summary: [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_concurrency_sweep_summary.txt](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_concurrency_sweep_summary.txt)
- Visualization:
  - TTFT chart: [ttft_vs_conc][ttft_vs_conc]
  - Throughput chart: [tok_s_vs_conc][tok_s_vs_conc]
- Action: publish concurrency guardrails for low-latency vs max-throughput modes.

### 6) WEIRD (transient, diagnosed): one-off `node2_gpu2` GEMM collapse did not persist
- What happened: a single run dropped to `~709 TFLOPS` on `node2_gpu2`; immediate isolated rerun recovered (`~1548.7 TFLOPS`), and current clean baseline remained in-family (`~1530.80 TFLOPS` on `node2_gpu2` avg).
- Why it matters: reset should be conditional, not default.
- Data:
  - Anomalous run: [results/structured/2026-02-08_ssh_key_full_suite_r2_node2_gemm_gpu_sanity.csv](results/structured/2026-02-08_ssh_key_full_suite_r2_node2_gemm_gpu_sanity.csv)
  - Immediate rerun: [results/structured/2026-02-08_node2_gpu2_diag_pre_reset_node2_gemm_gpu_sanity.csv](results/structured/2026-02-08_node2_gpu2_diag_pre_reset_node2_gemm_gpu_sanity.csv)
  - Clean baseline confirmation: [results/structured/2026-02-09_gb200_fullflags_all_0117_node2_gemm_gpu_sanity.csv](results/structured/2026-02-09_gb200_fullflags_all_0117_node2_gemm_gpu_sanity.csv)
- Visualization:
  - Transient anomaly chart: [docs/figures/2026-02-08_node2_gpu2_transient_gemm_tflops.png](docs/figures/2026-02-08_node2_gpu2_transient_gemm_tflops.png)
- Action: rerun isolated with locked clocks first; reset only if persistent.

### 7) NOTABLE: local scratch capacity exists but is unmounted by default
- What happened: multiple NVMe devices are present, but default benchmark path (`/tmp`) reflects root filesystem behavior.
- Why it matters: storage path decisions can dominate data staging and iteration time.
- Data:
  - Node1 storage: [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_storage.json](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_storage.json)
  - Node2 storage: [results/structured/2026-02-09_gb200_fullflags_all_0117_node2_storage.json](results/structured/2026-02-09_gb200_fullflags_all_0117_node2_storage.json)
  - fio baseline: [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_fio.json](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_fio.json)
- Visualization:
  - fio chart: [docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_fio.png](docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_fio.png)
  - Operator state snapshot: [operator_state_snapshot][operator_state_snapshot]
- Action: publish and automate a provider-approved `/scratch` policy.

### 8) NOTABLE: SHARP user-space present, but collective integration path is not operational
- What happened: forced NCCL CollNet checks failed before/after `sharp_am` start attempts.
- Why it matters: users cannot assume SHARP acceleration is available just because packages exist.
- Data:
  - SHARP check summary: [results/structured/2026-02-08_082000_ib_sharp_check_v3_ib_sharp_check.json](results/structured/2026-02-08_082000_ib_sharp_check_v3_ib_sharp_check.json)
  - CollNet failure excerpt: [results/structured/2026-02-08_082000_ib_sharp_check_v3_nccl_collnet_all_reduce_after_start_error_excerpt.txt](results/structured/2026-02-08_082000_ib_sharp_check_v3_nccl_collnet_all_reduce_after_start_error_excerpt.txt)
- Visualization:
  - Operator state snapshot: [operator_state_snapshot][operator_state_snapshot]
- Action: provider should document intended SHARP path (MPI HCOLL and/or NCCL net plugin) and health criteria.

### 9) WEIRD (operational reliability): orphan launchers can invalidate benchmark runs
- What happened: `/tmp` launcher scripts repeatedly started non-target runs during clean-run attempts.
- Why it matters: results can become invalid without obvious single-command failures.
- Data:
  - Launcher scripts evidence: [results/structured/2026-02-08_interference_launcher_scripts.txt](results/structured/2026-02-08_interference_launcher_scripts.txt)
  - Runtime/process snapshots: [results/structured/2026-02-08_interference_runtime_excerpts.txt](results/structured/2026-02-08_interference_runtime_excerpts.txt), [results/structured/2026-02-08_interference_process_snapshot.txt](results/structured/2026-02-08_interference_process_snapshot.txt)
- Visualization:
  - Operator state snapshot: [operator_state_snapshot][operator_state_snapshot]
- Action: enforce single queue runner + overlap detection as a hard policy.

## Implications For Small AI Teams
- Treat week-1 setup as operations-first:
  - lock in preflight policy (`persistenced`/`imex`/`dcgm`) before tuning kernels.
  - lock in launcher contract (OOB interface + NCCL socket/HCA envs) before scaling claims.
- Publish two explicit serving profiles from day one:
  - low-latency mode (`c<=256` where TTFT remains controlled),
  - max-throughput mode (`c=512` only when large TTFT is acceptable).
- Use parity checks as routine acceptance criteria:
  - node-level GEMM and NUMA parity looked healthy in the clean baseline,
  - include node2 fio in the next acceptance pass to close the remaining storage gap.
- Make queue discipline a hard rule (single runner + overlap detection) because hidden launch overlap can invalidate otherwise-clean benchmark runs.
- Treat observability as part of performance engineering:
  - DCGM lifecycle and restart policy directly affect benchmark validity and debug velocity.
- Keep "normal vs weird" artifacts retained; incident evidence is what turns one-off anomalies into actionable operator guidance.

## Stakeholder Recommendations (Prioritized)
1. `P0` Keep strict preflight mandatory: `nvidia-persistenced`, `nvidia-imex`, and `nvidia-dcgm` must be healthy before any benchmark/profiling run.
2. `P0` Keep DCGM as a hard requirement with before/after auditing; formalize whether `Restart=on-abort` is intentional and document expected behavior.
3. `P0` Publish a single multi-node launcher golden path (OOB/socket interface, HCA allowlist, port policy).
4. `P1` Publish serving guardrails: default concurrency envelopes for low-latency and max-throughput modes.
5. `P1` Publish provider storage policy for local NVMe scratch (`/scratch` design, durability expectations, lifecycle).
6. `P1` Clarify SHARP support stance and required software path (MPI HCOLL and/or NCCL plugin) with validation criteria.
7. `P1` Enforce single queue runner and overlap detection to prevent hidden benchmark contention.
8. `P2` Add continuous passive observability (log pipeline + alerting) to complement active suites.

## Capability Demonstration (Causal Debugging Workflow)
- Symptom detection: historical all-reduce entered a low-band regime (`~529.64 GB/s`) versus normal (`~840.55 GB/s`) under otherwise similar settings. Evidence: [results/structured/2026-02-07_224500_nccl_16g_baseline_ppr4_bindnone_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-07_224500_nccl_16g_baseline_ppr4_bindnone_node1node2_cluster_health_suite_summary.json), [results/structured/2026-02-07_140642_cluster_health_suite_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-07_140642_cluster_health_suite_node1node2_cluster_health_suite_summary.json), [docs/figures/2026-02-07_nccl_allreduce_bimodal_overlay.png](docs/figures/2026-02-07_nccl_allreduce_bimodal_overlay.png).
- Isolation: per-GPU and subset analysis identified physical `node1` GPU0 as the dominant bottleneck when included in collectives (historical investigation notes).
- Falsification: interface pinning, launch/mapping variants, and lock-vs-no-lock checks did not explain the full regime spread (historical investigation notes).
- Recovery path: targeted GPU reset and immediate revalidation restored normal compute/collective behavior in historical validation; this was treated as a fixable device/driver state, not a permanent capacity limit (historical investigation notes).
- Operator outcome: the reproducible workflow is detect -> isolate -> rerun locked -> escalate with targeted reset only if persistent; avoid blind node-wide resets.

## <cluster_perf_suite> Patchset
This case study produced a local patchset and harness extensions that demonstrate practical systems-debugging capability and operator-focused product improvement.

Patchset handling model (non-public source):
- Source diffs in `/home/ubuntu/ai-performance-engineering/code/<cluster_perf_suite>/` are treated as private and are not required in the public write-up.
- Public-facing deliverable uses capability/impact summaries plus reproducible evidence from `results/structured/` and `docs/figures/`.
- Reviewer handoff can be done via private repo access + commit hash/tag, while keeping this report artifact-focused.

| Patch area | Capability demonstrated | Operator impact |
| --- | --- | --- |
| `scripts/repro/run_vllm_serve_multinode_container.sh` + `scripts/repro/vllm_multinode_inner.sh` | First-class 2-node vLLM serving execution path with strict per-node clock-lock evidence and structured failure capture | Converts ad-hoc multinode serving attempts into reproducible artifacts (success or failure) suitable for field reports and operator handoff. |
| `analysis/plot_nvlink_topology.py` + suite integration in `scripts/run_cluster_eval_suite.sh` | Topology artifact productization from discovery metadata (`nvidia-smi topo -m`) | Adds explicit NVLink/NVSwitch evidence to stakeholder packages without requiring a separate manual extraction workflow. |
| `.gitignore` | Output hygiene and reproducibility policy enforcement (generated vLLM result artifacts ignored by default) | Keeps review/commit surface focused on code and documented artifacts, not transient run outputs. |
| `standalone/compute/run-all-benchmarks.sh` | Container/runtime robustness debugging (GPU access check path hardening) | Reduces first-run failures from image/tag assumptions and improves setup reliability. |
| `standalone/compute/nvbandwidth/run-nvbandwidth.sh` | Build-chain hardening for benchmark dependencies | Improves portability across base images; fewer failed compute diagnostics. |
| `standalone/compute/p2p-bandwidth/p2p-bandwidth.py` | Measurement-correctness validation (timing path reliability) | Avoids misleading bandwidth claims from unstable timing methodology. |
| `standalone/compute/gemm-bench/grouped_gemm_bench.py` + `code/cluster_perf_patches/deepgemm_gb200_grouped_gemm_ue8m0.patch` | Architecture-specific compatibility triage on GB200 | Restores reproducible FP4 grouped-GEMM behavior where legacy scaling-factor errors appeared. |
| `standalone/storage/fio/run-fio-bench.sh` | Failure-mode fallback design (native path when container pull fails) | Maintains storage benchmark continuity under registry/auth constraints. |
| `standalone/inference/vllm/run-vllm-bench.sh` | Serving runbook hardening (health/smoke/logging robustness) | Faster diagnosis and fewer silent failures in inference runs. |
| `standalone/inference/vllm/run-vllm-bench-multinode.sh` | Multinode inference launch workflow prototyping for GB200 trays | Provides a concrete operator entrypoint for two-node serving validation in environments without Slurm/K8s orchestration. |
| `standalone/networking/allreduce/all_reduce_bench.py` | Guardrail engineering for environment mismatches | Fail-fast behavior improves debuggability when CUDA is unavailable/misconfigured. |
| `standalone/networking/allreduce/run_2node_bench.sh` | Simplified two-node collective launch script for operator use | Reduces friction for reproducing multinode all-reduce checks from first contact. |
| `standalone/docs/gb200-networking.md` | Operator documentation synthesis for multinode reality | Makes known-good launch path and required constraints explicit for reproducibility. |

Evidence that these improvements are usable in practice is captured by the extended harness outputs and figures, including:
- `results/structured/2026-02-08_gb200_fullflags_all_233428_manifest.json`
- `results/structured/2026-02-08_gb200_fullflags_all_233428_nccl_algo_comparison.json`
- `results/structured/2026-02-08_gb200_fullflags_all_233428_allreduce_stability.json`
- `results/structured/2026-02-08_gb200_fullflags_all_233428_allreduce_latency_comp.json`
- `results/structured/2026-02-08_gb200_fullflags_all_233428_allgather_control_plane.json`
- `results/structured/2026-02-08_gb200_fullflags_all_233428_node1_multinode_torchrun_train_step.json`
- `results/structured/2026-02-08_gb200_fullflags_all_233428_node1_checkpoint_io.json`
- `docs/figures/2026-02-08_gb200_fullflags_all_233428_nccl_algo_comparison.png`
- `docs/figures/2026-02-08_gb200_fullflags_all_233428_allreduce_stability.png`
- `docs/figures/2026-02-08_gb200_fullflags_all_233428_node1_multinode_torchrun_train_step.png`
- `results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.json`
- `results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.csv`
- `results/structured/2026-02-09_gb200_fullflags_all_0117_node1_nvlink_topology.json`
- `results/structured/2026-02-09_gb200_fullflags_all_0117_node2_nvlink_topology.json`
- `docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_nvlink_topology.png`
- `docs/figures/2026-02-09_gb200_fullflags_all_0117_node2_nvlink_topology.png`

## Repro Steps
Canonical command used for the baseline run (`2026-02-09_gb200_fullflags_all_0117`):
```bash
cd code/cluster

scripts/run_cluster_eval_suite.sh \
  --run-id 2026-02-09_gb200_fullflags_all_0117 \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --oob-if enP22p3s0f3 \
  --socket-ifname enP22p3s0f3 \
  --nccl-ib-hca mlx5_0,mlx5_1,mlx5_4,mlx5_5 \
  --health-suite extended \
  --health-gdr \
  --health-gdr-gpu 0 \
  --health-gdr-mem-types 0,1 \
  --health-gdr-use-dmabuf \
  --fp4-suite-dir /home/ubuntu/ai-performance-engineering/code/clustermax \
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

For fresh reruns, keep the same flags and only change `--run-id`.

Optional follow-on commands used for the new stakeholder artifacts:
```bash
# Multinode vLLM serving path (strict lock + structured outputs)
scripts/repro/run_vllm_serve_multinode_container.sh \
  --run-id 2026-02-09_gb200_fullflags_all_0117 \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --model openai/gpt-oss-120b \
  --tp 8 \
  --isl 512 \
  --osl 256 \
  --concurrency 16 \
  --num-prompts 64 \
  --socket-ifname enP22p3s0f3 \
  --nccl-ib-hca mlx5_0,mlx5_1,mlx5_4,mlx5_5

# NVLink/NVSwitch topology artifacts from node meta files
python3 analysis/plot_nvlink_topology.py \
  --meta results/structured/2026-02-09_gb200_fullflags_all_0117_node1_meta.json \
  --fig-out docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_nvlink_topology.png \
  --summary-out results/structured/2026-02-09_gb200_fullflags_all_0117_node1_nvlink_topology.json

python3 analysis/plot_nvlink_topology.py \
  --meta results/structured/2026-02-09_gb200_fullflags_all_0117_node2_meta.json \
  --fig-out docs/figures/2026-02-09_gb200_fullflags_all_0117_node2_nvlink_topology.png \
  --summary-out results/structured/2026-02-09_gb200_fullflags_all_0117_node2_nvlink_topology.json
```

## `--disable-fp4` if needed
- FP4 checks are enabled by default in the suite.
- Use for portability run to avoid requiring external FP4 suite/image dependencies.
- Suggestion: Use FP4-enabled runs as an explicit second pass when the dependency chain is available.

## Local Patch Prerequisite (for FP4-Enabled Repro)
FP4 grouped-GEMM reproducibility on GB200 requires the local patch:
`code/cluster_perf_patches/deepgemm_gb200_grouped_gemm_ue8m0.patch`.
Without it, grouped DeepGEMM may report
`Unsupported architecture or scaling factor types`.

Apply steps (exact):
```bash
cd /home/ubuntu/ai-performance-engineering/code/cluster

export SUITE_ROOT=/path/to/cluster_perf_suite
export TARGET="${SUITE_ROOT}/standalone/compute/gemm-bench/grouped_gemm_bench.py"
export PATCH_SRC="code/cluster_perf_patches/deepgemm_gb200_grouped_gemm_ue8m0.patch"
export PATCH_TMP="/tmp/deepgemm_gb200_grouped_gemm_ue8m0.${USER}.patch"

test -f "${TARGET}"
cp "${TARGET}" "${TARGET}.pre_ue8m0.bak"

sed \
  -e "s|^--- code/cluster_perf_suite_snapshot/standalone/compute/gemm-bench/grouped_gemm_bench.py$|--- ${TARGET}|" \
  -e "s|^+++ code/cluster_perf_suite/standalone/compute/gemm-bench/grouped_gemm_bench.py$|+++ ${TARGET}|" \
  "${PATCH_SRC}" > "${PATCH_TMP}"

patch --dry-run -p0 < "${PATCH_TMP}"
patch -p0 < "${PATCH_TMP}"
```

Verification checks:
```bash
rg -n \
  "use_ue8m0 = arch_major >= 10|disable_ue8m0_cast = not use_ue8m0|m_grouped_fp8_gemm_nt_contiguous|DeepGEMM unsupported|per_token_cast_to_fp8\\(a_bf16, use_ue8m0=use_ue8m0\\)|per_block_cast_to_fp8\\(b_bf16\\[i\\], use_ue8m0=use_ue8m0\\)" \
  "${TARGET}"

scripts/run_cluster_perf_grouped_gemm.sh \
  --suite-dir "${SUITE_ROOT}" \
  --run-id <run_id> \
  --label <label> \
  --image <image> \
  --preset auto \
  --warmup 2 \
  --iters 5

python3 - <<'PY' "results/structured/<run_id>_<label>_cluster_perf_grouped_gemm_summary.json"
import json, sys
summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
reason = ((summary.get("deepgemm") or {}).get("unsupported_reason") or "")
if "Unsupported architecture or scaling factor types" in reason:
    raise SystemExit("FAIL: legacy DeepGEMM scaling-factor error still present")
print("OK: summary does not contain the legacy scaling-factor error")
PY
```

## Reproducibility Package
- Clean baseline manifest: [results/structured/2026-02-09_gb200_fullflags_all_0117_manifest.json](results/structured/2026-02-09_gb200_fullflags_all_0117_manifest.json)
- Sanitized cluster metadata aggregator: [latest_cluster_meta][latest_cluster_meta]
- Clean preflight (DCGM before/after): [preflight_latest][preflight_latest]
- Clean health summary: [latest_health][latest_health]
- Multinode vLLM path artifact (strict-lock + structured failure package): [results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.json](results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.json)
- NVLink/NVSwitch topology artifacts: [node1_nvlink_topology_json][node1_nvlink_topology_json], [node2_nvlink_topology_json][node2_nvlink_topology_json]
- Manifest summary: `286` files (`106 json`, `23 csv`, `4 jsonl`, `26 png`, `30 txt`, `97 log`).
- Historical incident bundle (only what changed decisions):
  - Historical run (`2026-02-08_032814_cloud_eval_full_fixed`) service/context snapshot: [results/structured/2026-02-08_032814_cloud_eval_full_fixed_preflight_services.json](results/structured/2026-02-08_032814_cloud_eval_full_fixed_preflight_services.json)
  - NVLS failure excerpt: [results/structured/2026-02-08_025442_cloud_eval_full_health_suite_extended_node1node2_nccl_all_reduce_perf.error_excerpt.txt](results/structured/2026-02-08_025442_cloud_eval_full_health_suite_extended_node1node2_nccl_all_reduce_perf.error_excerpt.txt)
  - NVLS-off tradeoff run: [results/structured/2026-02-08_031531_health_suite_extended_nvls0_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-08_031531_health_suite_extended_nvls0_node1node2_cluster_health_suite_summary.json)

## Repository Handoff (GitHub)
- Repository URL: `git@github.com:cfregly/ai-performance-engineering.git`
- Commit for review (current local HEAD): `d58da6a469df535849e08c1aba2d3ea46c542ec1`
- Collaborator access (`JordanNanos`) status: not recorded in this artifact package; requires explicit owner confirmation during handoff.

## Appendix

### Monitoring Expectations Coverage

| Expectation | Status | Notes / Evidence |
| --- | --- | --- |
| Cluster overview (nodes/workloads/health trends) | PARTIAL | Node-level health is covered by active suite artifacts; no control-plane dashboard in scope. [latest_health] |
| Control-plane health (K8s API/etcd/scheduler) | MISSING | No K8s control plane observed in scope. [latest_cluster_meta] |
| Slurm integration and job stats | MISSING | No Slurm control-plane evidence in scope. [latest_cluster_meta] |
| Kubernetes metrics stack (`kube-prometheus`, `kube-state-metrics`, `node-exporter`, `cAdvisor`) | MISSING | Not observed in scope. [latest_cluster_meta] |
| DCGM exporter/hostengine reliability | PARTIAL | Now hard-required in preflight with before/after auditing; provider restart policy remains a decision point. [preflight_historical_dcgm] [preflight_latest] |
| KV-cache metrics for HPA (`gpu_cache_usage_perc`) | MISSING | No K8s/HPA integration in scope. |
| Alerting/notification integration | UNKNOWN | Provider alerting surface not evaluated in this SSH-only scope. |
| Node power/thermal telemetry | PARTIAL | Node-level GPU telemetry is captured; no fleet dashboard in scope. [latest_cluster_meta] |
| PCIe AER monitoring | MISSING | Not collected as a first-class metric in this runbook. |
| dmesg/log pipeline (promtail or equivalent) | PARTIAL | Incident-focused kernel evidence was captured during investigation, but no continuous log pipeline was evaluated in this package. |
| TFLOPs/SM active/occupancy via DCGM profiling counters | MISSING | Not captured in this evaluation package. |
| Nsight Compute availability for users | UNKNOWN | Not fully validated as a managed user-facing workflow here. |
| NVLink/XGMI throughput visibility | PARTIAL | Dedicated NVLink/NVSwitch topology artifacts are now bundled (`node1/node2`), but a dedicated throughput microbenchmark bundle is still not included. [node1_nvlink_topology_json] [node2_nvlink_topology_json] |
| PCIe host<->GPU throughput visibility | PARTIAL | Capability was checked during investigation, but dedicated `nvbandwidth` artifacts are not bundled in this report package. |
| InfiniBand/RoCE throughput visibility | YES | IB perftest + NCCL multi-node suite evidence captured. [latest_health] |
| User/group quotas and scheduler history | MISSING | No scheduler resource governance surface in scope. |
| Active + passive health-check integration | PARTIAL | Active checks are strong; passive continuous monitoring was not in scope. [latest_health] [preflight_latest] |

### What Historical Runs Changed (Why They Are Kept)
- They proved causal links for operational failures that the clean run no longer shows.
- They quantified the real mitigation cost (`NVLS off` reduced all-reduce peak from `~839.39` to `~699.63 GB/s`).
- They justified hard preflight gating and explicit fallback reporting in the harness.
- They prevented over-claiming from a single “all good” run.

### Open Questions for Provider
- Is `nvidia-dcgm` `Restart=on-abort` an intentional SRE policy?
- What is the expected SHARP enablement path for user collectives on this image?
- What is the intended lifecycle and performance policy for local NVMe scratch?
- Which control-plane/scheduler observability surface should users rely on (if any) in this environment?

## Activity Log
<!-- ACTIVITY_LOG_START -->
- 2026-02-09: Executed full multi-node suite with GB200-focused flags under `RUN_ID=2026-02-09_gb200_fullflags_all_0117`; driver status `0` and suite summary `STATUS: OK`.
- 2026-02-09: Validated complete package generation (`manifest.json` + structured outputs + figures): manifest file count `286` with hashes for all artifacts.
- 2026-02-09: Added sanitized cluster metadata aggregator `results/structured/2026-02-09_gb200_fullflags_all_0117_cluster_meta.json` and linked it as the single metadata reference in report sections.
- 2026-02-09: Fixed MAMF concurrent-per-GPU clock-lock mapping in `scripts/run_mamf_finder_all_nodes.sh` (set lock target to logical device `0` when `CUDA_VISIBLE_DEVICES` is pinned), removing `Invalid device id` failures.
- 2026-02-09: Re-ran MAMF on both nodes (`8/8` GPUs) with the fixed path and refreshed `docs/figures/2026-02-09_gb200_fullflags_all_0117_mamf_straggler.png`.
- 2026-02-09: Added unified dashboard artifact `docs/figures/2026-02-09_gb200_fullflags_all_0117_cluster_story_dashboard.png` and structured node parity summary `results/structured/2026-02-09_gb200_fullflags_all_0117_node_parity_summary.json`.
- 2026-02-09: Expanded first-contact timeline and operator guidance in this report with concrete UTC timestamps from suite logs.
- 2026-02-09: Added native multi-node vLLM harness path (`scripts/repro/run_vllm_serve_multinode_container.sh`) with strict lock metadata on both nodes and structured failure outputs.
- 2026-02-09: Executed the multi-node vLLM path for `RUN_ID=2026-02-09_gb200_fullflags_all_0117`; run failed before server readiness with `ModuleNotFoundError: vllm.config.kernel`, captured under `results/structured/2026-02-09_gb200_fullflags_all_0117_node1_vllm_multinode_serve.json`.
- 2026-02-09: Added dedicated NVLink/NVSwitch topology artifacts for both nodes (`*_nvlink_topology.json` + `*_nvlink_topology.png`) from captured `nvidia-smi topo -m` metadata.
<!-- ACTIVITY_LOG_END -->

---

[operator_state_snapshot]: docs/figures/2026-02-08_operator_state_snapshot.png
[tok_s_vs_conc]: docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_vllm_serve_total_tok_s_vs_concurrency.png
[ttft_vs_conc]: docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_vllm_serve_ttft_vs_concurrency.png
[cluster_story_dashboard]: docs/figures/2026-02-09_gb200_fullflags_all_0117_cluster_story_dashboard.png
[node_parity_summary]: results/structured/2026-02-09_gb200_fullflags_all_0117_node_parity_summary.json
[node1_nvlink_topology_json]: results/structured/2026-02-09_gb200_fullflags_all_0117_node1_nvlink_topology.json
[node2_nvlink_topology_json]: results/structured/2026-02-09_gb200_fullflags_all_0117_node2_nvlink_topology.json
[node1_nvlink_topology_png]: docs/figures/2026-02-09_gb200_fullflags_all_0117_node1_nvlink_topology.png
[node2_nvlink_topology_png]: docs/figures/2026-02-09_gb200_fullflags_all_0117_node2_nvlink_topology.png
[latest_cluster_meta]: results/structured/2026-02-09_gb200_fullflags_all_0117_cluster_meta.json
[latest_health]: results/structured/2026-02-09_gb200_fullflags_all_0117_health_suite_extended_node1node2_cluster_health_suite_summary.json
[preflight_latest]: results/structured/2026-02-09_gb200_fullflags_all_0117_preflight_services.json
[preflight_historical_dcgm]: results/structured/2026-02-08_test_preflight_dcgm_before_after_node1node2_preflight_services.json
