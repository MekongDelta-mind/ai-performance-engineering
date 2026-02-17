# Cluster Perf Field Report (GB200, 2 Nodes)

Last updated: 2026-02-10. Canonical run: `2026-02-10_full_suite_e2e_wire_qf_mon`.

## Table of Contents
1. [TL;DR](#tldr)
2. [Scope + Canonical Artifacts](#scope--canonical-artifacts)
3. [Required Reliability Gates (Canonical Run)](#required-reliability-gates-canonical-run)
4. [Operator Friction + Monitoring Expectations (New Checks)](#operator-friction--monitoring-expectations-new-checks)
5. [TL;DR Evidence Anchors](#tldr-evidence-anchors)
6. [Cluster Story (First Contact)](#cluster-story-first-contact)
7. [Weird / New / Interesting (with Normal Baseline)](#weird--new--interesting-with-normal-baseline)
8. [Benchmark A (Networking Story)](#benchmark-a-networking-story)
9. [Benchmark B (Inference Story)](#benchmark-b-inference-story)
10. [Node Parity Snapshot (node1 vs node2)](#node-parity-snapshot-node1-vs-node2)
11. [Per-Node Deep-Dive Visuals (Restored)](#per-node-deep-dive-visuals-restored)
12. [NVLink/NVSwitch Topology Snapshot](#nvlinknvswitch-topology-snapshot)
13. [Dedicated nvbandwidth Snapshot](#dedicated-nvbandwidth-snapshot)
14. [GB200 Extensions (Enabled in Canonical Run)](#gb200-extensions-enabled-in-canonical-run)
15. [Required Issues (Explicit)](#required-issues-explicit)
16. [Root Cause + Fix Mapping](#root-cause--fix-mapping)
17. [Report Completeness Delta (vs prior condensed revision)](#report-completeness-delta-vs-prior-condensed-revision)
18. [Gaps, Risks, and Smell Checks](#gaps-risks-and-smell-checks)
19. [Implications for Small AI Teams](#implications-for-small-ai-teams)
20. [Stakeholder Recommendations (Prioritized)](#stakeholder-recommendations-prioritized)
21. [Repro Steps](#repro-steps)
22. [Reproducibility Package](#reproducibility-package)
23. [Activity Log](#activity-log)
24. [Appendix (Coverage vs Case-Study Goals)](#appendix-coverage-vs-case-study-goals)

## TL;DR
| Topic | Summary |
| --- | --- |
| Scope | In-scope hosts: `node1`, `node2`; 4x GB200 per host; excluded nodes: none. |
| Canonical run | `2026-02-10_full_suite_e2e_wire_qf_mon` |
| Required reliability gates | In canonical run, all 3 required gates are green (`hang_triage_bundle`, `connectivity_probe`, `nccl_env_sensitivity`). |
| Suite health | `53/53` suite steps green; `validate_required_artifacts=0`; no remediation-only state required for canonical package. |
| Networking headline | NCCL all-reduce peak `839.17 GB/s`; torch distributed all-reduce peak `719.16 GB/s`; IB write BW `387.14 / 387.13 / 387.13 / 387.13 Gbps`; OOB TCP `7.660 / 7.450 Gbps` (fwd/rev). |
| Inference headline | Single-node vLLM peaks at `54,947.294 tok/s` (`c=256`) with p99 TTFT `869.511 ms`; by `c=512`, mean/p99 TTFT rise to `7,004.857 / 11,955.548 ms` (severe latency knee). |
| Multinode inference | Multinode vLLM artifact is present and clean: `c=64`, `16,679.879 tok/s`, p99 TTFT `1,260.348 ms`, status `ok`. |
| Operator checks (new) | `quick_friction` and `monitoring_expectations` both executed for node1+node2 in canonical run; each host is `degraded` with explicit missing-tool diagnostics captured. |
| Required issue closure | `node2_fio.json` present; multinode vLLM artifacts present; nvbandwidth bundle present on both nodes; health-suite GDR `requested=true`, `effective_enabled=true`. |
| Remaining risk | High-concurrency serving tail latency is the main user-facing risk; treat `c=512` as throughput mode, not low-latency mode. |

## Scope + Canonical Artifacts
| Item | Value |
| --- | --- |
| Hosts in-scope | `node1,node2` |
| Excluded hosts | none |
| GPUs per host | 4 |
| Canonical manifest | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_manifest.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_manifest.json) |
| Canonical suite steps | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json) |
| Discovery/meta | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_meta.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_meta.json) and [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_meta.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_meta.json) |
| Health summary | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json) |
| Node parity summary | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node_parity_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node_parity_summary.json) |
| Manifest summary counts | `543` files (`193 json`, `26 csv`, `4 jsonl`, `244 log`, `33 png`, `40 txt`, `1 tsv`, `2 py`) |

## Required Reliability Gates (Canonical Run)
These gates were executed in the canonical full-suite run and all completed successfully on in-scope hosts (`node1,node2`).

| Gate | Status | Key result | Structured artifact |
| --- | --- | --- | --- |
| Hang-triage readiness (`py-spy` + `strace`) | `ok` on both hosts | `py-spy 0.4.1`, `strace 6.8`, semantic status `ok` on node1+node2 | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_hang_triage_readiness.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_hang_triage_readiness.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_hang_triage_readiness.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_hang_triage_readiness.json) |
| Torchrun connectivity probe | `ok` | `world_size=8`, barrier mean `0.137 ms` (`p95=0.230 ms`), payload probe busbw range `0.057134-0.057147 Gbps` | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_torchrun_connectivity_probe.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_torchrun_connectivity_probe.json) |
| NCCL env sensitivity sweep | `ok` (`failure_count=0`) | Baseline peak `417.57 GB/s`; best profile `baseline_auto` at `417.57 GB/s` (`1.000000x`) | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.json) |

| NCCL profile | Peak bus BW (GB/s) | Speedup vs baseline |
| --- | ---: | ---: |
| baseline_auto | `417.57` | `1.000000x` |
| cross_nic_0 | `415.85` | `0.995881x` |
| cross_nic_1 | `416.99` | `0.998611x` |
| cross_nic_2 | `415.56` | `0.995186x` |
| qps_4 | `416.13` | `0.996551x` |
| ctas_16_16 | `417.22` | `0.999162x` |

<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.png" alt="NCCL env sensitivity canonical gate validation" width="920"/></a></p>

Data: [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_manifest.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_manifest.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_hang_triage_readiness.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_hang_triage_readiness.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_hang_triage_readiness.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_hang_triage_readiness.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_torchrun_connectivity_probe.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_torchrun_connectivity_probe.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.json)

Raw logs: [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node1_hang_triage_readiness.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node1_hang_triage_readiness.log), [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node2_hang_triage_readiness.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node2_hang_triage_readiness.log), [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_torchrun_connectivity_probe_node0.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_torchrun_connectivity_probe_node0.log), [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_torchrun_connectivity_probe_node1.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_torchrun_connectivity_probe_node1.log), [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_cross_nic_2_nccl_env_cross_nic_2_nccl_all_reduce.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_cross_nic_2_nccl_env_cross_nic_2_nccl_all_reduce.log)

## Operator Friction + Monitoring Expectations (New Checks)
Both checks now run in canonical flow and are included as required stakeholder evidence.

| Check | node1 | node2 | Key diagnostics | Structured artifacts |
| --- | --- | --- | --- | --- |
| Quick friction battery | `degraded` (`4/7` pass) | `degraded` (`4/7` pass) | Missing tools on both nodes: `huggingface-cli`, `whois`, `speedtest-cli/speedtest`; install + NGC pull checks are green. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_quick_friction.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_quick_friction.json) |
| Monitoring expectations snapshot | `degraded` | `degraded` | `gpu_telemetry` category is fully `ok`; `control_plane` missing (`kubectl` absent) and `system_signals` missing (`dmesg` permission denied). | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_monitoring_expectations.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_monitoring_expectations.json) |

Raw logs: [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.log), [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node2_quick_friction.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node2_quick_friction.log), [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.log), [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node2_monitoring_expectations.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node2_monitoring_expectations.log)

## TL;DR Evidence Anchors
| Claim | Data | Visual |
| --- | --- | --- |
| Suite is fully green and canonical package is complete. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json) | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png) |
| Networking fabric is strong and stable at large message sizes. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json) | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_2nodes_nccl_bw_vs_msg.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_2nodes_nccl_bw_vs_msg.png) |
| vLLM throughput climbs with concurrency, but TTFT knee is severe at high concurrency. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv) | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_ttft_vs_concurrency.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_ttft_vs_concurrency.png) |
| Multinode vLLM path is now captured and clean. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.json) | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_vllm_serve_total_tok_s_vs_concurrency.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_vllm_serve_total_tok_s_vs_concurrency.png) |
| nvbandwidth bundle exists on both nodes and runs in host runtime with compat libs injected (root-cause fix, not fallback runtime switching). | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json) | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth_sums.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth_sums.png) |
| New operator checks are executed and diagnosable in canonical output. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json) | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png) |

## Cluster Story (First Contact)
| UTC time | Milestone | Status |
| --- | --- | --- |
| `2026-02-10T06:22:18Z` | `bootstrap_nodes` started | ok |
| `2026-02-10T06:23:14Z` | `preflight_services` started | ok |
| `2026-02-10T06:35:23Z` | single-node NCCL started | ok |
| `2026-02-10T06:35:35Z` | multi-node NCCL started | ok |
| `2026-02-10T06:37:22Z` | extended health suite started | ok |
| `2026-02-10T06:43:06Z` | vLLM single-node sweep started | ok |
| `2026-02-10T06:56:33Z` | vLLM multinode serve (`c=64`) started | ok |
| `2026-02-10T07:32:45Z` | nvbandwidth all-nodes bundle started | ok |
| `2026-02-10T07:38:09Z` | required-artifact validation + manifest refresh | ok |

Interpretation: time-to-first-multinode signal was fast (NCCL in under 2 minutes from bootstrap start), then the long poles were serving sweeps and multinode serving stabilization.

<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png" alt="Cluster story dashboard" width="920"/></a></p>

Data: [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json)

## Weird / New / Interesting (with Normal Baseline)
### Baseline vs Weird Log
| Area | Normal (canonical) | Weird / notable | Evidence |
| --- | --- | --- | --- |
| NCCL all-reduce | Peak `839.17 GB/s`, stable high-band regime | Stability run still shows moderate jitter (`CV=2.189%`, min outliers down to `678.603 GB/s`) | [health summary](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json), [allreduce stability](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_stability.json) |
| Service-state gating | `persistenced`/`imex`/`dcgm` are active on both nodes before health/benchmark execution | This remains a hard validity gate; any service-state drift must invalidate runs, not degrade silently | [preflight services](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_preflight_services.json), [health preflight services](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_preflight_services.json) |
| GDR path | `requested=true`, `effective_enabled=true`, tags include `gdr_gpu0_mem0` and `gdr_gpu0_mem0_dmabuf` | Requested mem-type matrix is intentionally constrained to supported mem type (`0`); unsupported modes are treated as preflight failures | [health summary](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json), [health log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_suite/health_suite_extended.log) |
| Serving behavior | Throughput scales to `54,947.294 tok/s` (single-node, `c=256`) and multinode path is clean (`status=ok`) | Tail latency knee is severe at `c=512` (mean TTFT `7,004.857 ms`, p99 `11,955.548 ms`) | [serve sweep csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv), [multinode json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.json) |
| nvbandwidth path | Node1 + node2 bundles present, status `ok`, effective runtime `host` | Host path requires explicit CUDA compat-lib chain due historical user-mode mismatch | [node1 nvbandwidth json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json), [node2 nvbandwidth json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json) |
| Node parity | Node-level GEMM means are tight (`node2/node1=0.988x`) and both fio artifacts are present | GPU-level straggler spread remains non-trivial (`10.41%` gap-to-best, `11.62%` min-to-max) and fio random-write asymmetry persists (`node2/node1=0.917x`) | [node parity summary](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node_parity_summary.json), [node1 fio](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_fio.json), [node2 fio](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_fio.json), [node1 gpu0 mamf](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_gpu0_mamf_summary.json), [node1 gpu2 mamf](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_gpu2_mamf_summary.json) |
| Launch ergonomics | Time-to-first multi-node signal is fast (~13s from single-node NCCL start to multi-node NCCL start) | Wall-clock is dominated by serving sweeps and multinode stabilization, so schedule budgeting must account for long poles | [suite steps](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json) |

### Deep-Dive Findings
| Finding | Baseline anchor | Reinforcement insight | Evidence |
| --- | --- | --- | --- |
| 1. Reliability is now service-gated by default | `Service-state gating` | Run validity is tied to explicit preflight state capture; this is now operationally auditable, not implicit. | [preflight services](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_preflight_services.json), [health preflight services](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_preflight_services.json) |
| 2. GDR coverage is explicit and constrained | `GDR path` | Canonical flow prioritizes supported-mode correctness over broad-mode ambiguity; unsupported mem-types are not silently downgraded. | [health summary](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json), [health log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_suite/health_suite_extended.log) |
| 3. Throughput and latency goals diverge sharply | `Serving behavior` | Peak throughput happens at `c=256`; `c=512` is overflow throughput mode, not interactive mode. | [serve sweep csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv), [multinode json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.json) |
| 4. Root-cause fix quality matters | `nvbandwidth path` | Canonical host-runtime success is now tied to compat-lib chain correctness, not runtime fallback behavior. | [node1 nvbandwidth](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json), [node2 nvbandwidth](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json) |
| 5. Parity and straggler signals must both be tracked | `Node parity` | Node means alone are insufficient; per-GPU spread plus degraded operator checks should be trended together. | [node parity summary](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node_parity_summary.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json) |

<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png" alt="Cluster story timeline and long poles" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_stability.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_stability.png" alt="All-reduce stability (jitter context)" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_ttft_vs_concurrency.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_ttft_vs_concurrency.png" alt="Single-node vLLM TTFT knee" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth_sums.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth_sums.png" alt="nvbandwidth sums (root-cause fix path)" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_mamf_straggler.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_mamf_straggler.png" alt="MAMF straggler spread" width="920"/></a></p>

Data: [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_stability.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_stability.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_gpu0_mamf_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_gpu0_mamf_summary.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_gpu2_mamf_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_gpu2_mamf_summary.json)

## Benchmark A (Networking Story)
| Metric | Value |
| --- | ---: |
| NCCL all-reduce peak bus bandwidth | `839.17 GB/s` |
| NCCL all-gather peak bus bandwidth | `656.16 GB/s` |
| NCCL reduce-scatter peak bus bandwidth | `675.89 GB/s` |
| NCCL alltoall peak bus bandwidth | `604.13 GB/s` |
| torch distributed all-reduce peak bus bandwidth | `719.16 GB/s` |
| IB write BW (`mlx5_0/1/4/5`) | `387.14 / 387.13 / 387.13 / 387.13 Gbps` |
| OOB TCP (fwd/rev) | `7.660 / 7.450 Gbps` |

Interpretation: fabric is healthy and high-band; control plane remains much slower than data plane.

<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_2nodes_nccl_bw_vs_msg.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_2nodes_nccl_bw_vs_msg.png" alt="2-node NCCL bus bandwidth vs message size" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_2nodes_nccl_scaling_efficiency.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_2nodes_nccl_scaling_efficiency.png" alt="2-node NCCL scaling efficiency" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_iperf3_oob_tcp.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_iperf3_oob_tcp.png" alt="OOB TCP throughput" width="920"/></a></p>

Data: [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_2nodes_nccl.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_2nodes_nccl.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_iperf3_oob_tcp.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_iperf3_oob_tcp.json)

## Benchmark B (Inference Story)
### Single-node sweep
| Concurrency | Total tok/s | Mean TTFT (ms) | p99 TTFT (ms) | p99 TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| `32` | `12921.701` | `233.299` | `498.527` | `4.969` |
| `64` | `25426.417` | `189.140` | `460.626` | `6.004` |
| `128` | `36526.639` | `305.124` | `789.292` | `8.001` |
| `256` | `54947.294` | `425.749` | `869.511` | `10.929` |
| `512` | `51328.705` | `7004.857` | `11955.548` | `19.081` |

### Multinode sweep
| Concurrency | Total tok/s | Mean TTFT (ms) | p99 TTFT (ms) | p99 TPOT (ms) | Status |
| ---: | ---: | ---: | ---: | ---: | --- |
| `64` | `16679.879` | `257.364` | `1260.348` | `7.589` | `ok` |

Interpretation: throughput keeps rising, but TTFT tails become unacceptable for interactive latency goals at high concurrency.

<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_total_tok_s_vs_concurrency.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_total_tok_s_vs_concurrency.png" alt="Single-node vLLM throughput vs concurrency" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_ttft_vs_concurrency.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_ttft_vs_concurrency.png" alt="Single-node vLLM TTFT vs concurrency" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_vllm_serve_ttft_vs_concurrency.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_vllm_serve_ttft_vs_concurrency.png" alt="Multinode vLLM TTFT vs concurrency" width="920"/></a></p>

Data: [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.csv)

## Node Parity Snapshot (node1 vs node2)
| Metric | node1 | node2 | node2/node1 |
| --- | ---: | ---: | ---: |
| GEMM mean TFLOPS | `1544.084` | `1524.918` | `0.988x` |
| GEMM min TFLOPS | `1480.291` | `1503.477` | `1.016x` |
| NUMA local BW (GB/s) | `136.605` | `135.850` | `0.994x` |
| fio seq read (MB/s) | `1389.928` | `1395.214` | `1.004x` |
| fio seq write (MB/s) | `714.948` | `765.431` | `1.071x` |
| fio rand read IOPS | `43316.981` | `40191.841` | `0.928x` |
| fio rand write IOPS | `18590.278` | `17054.312` | `0.917x` |

Interpretation: compute and NUMA are tightly aligned; storage asymmetry is moderate and should be trended.

<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_gemm_gpu_sanity.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_gemm_gpu_sanity.png" alt="GEMM parity chart" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_fio.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_fio.png" alt="Node1 fio" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node2_fio.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node2_fio.png" alt="Node2 fio" width="920"/></a></p>

Data: [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node_parity_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node_parity_summary.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_fio.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_fio.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_fio.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_fio.json)

## Per-Node Deep-Dive Visuals (Restored)
| Visual bundle | Why included | Data |
| --- | --- | --- |
| Node-level NCCL scaling (`node1`) | Single-node collective scaling context for the 2-node story. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nccl.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nccl.json) |
| Node-level NUMA bandwidth (`node1`,`node2`) | Confirms local-memory parity context behind node-to-node compute parity. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_numa_mem_bw.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_numa_mem_bw.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_numa_mem_bw.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_numa_mem_bw.json) |
| Train-step per-mode curves (`single` vs `multinode`) | Shows train-step behavior used by scaling summary in GB200 extensions. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_single_node_torchrun_train_step.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_single_node_torchrun_train_step.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_torchrun_train_step.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_torchrun_train_step.json) |
| vLLM TPOT curves (`single` vs `multinode`) | Adds TPOT shape context beyond throughput and TTFT charts. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.csv) |
| C2C memcpy micro-shapes (`node1`) | Separates bandwidth and latency shape effects inside C2C snapshot. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_c2c_memcpy.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_c2c_memcpy.json) |
| Per-node grouped GEMM charts (`node1`,`node2`) | Complements parity table with direct per-node grouped GEMM visuals. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_cluster_perf_grouped_gemm_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_cluster_perf_grouped_gemm_summary.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_cluster_perf_grouped_gemm_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_cluster_perf_grouped_gemm_summary.json) |

<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nccl_bw_vs_msg.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nccl_bw_vs_msg.png" alt="Node1 NCCL bus bandwidth vs message size" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nccl_scaling_efficiency.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nccl_scaling_efficiency.png" alt="Node1 NCCL scaling efficiency" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_numa_mem_bw.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_numa_mem_bw.png" alt="Node1 NUMA memory bandwidth" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node2_numa_mem_bw.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node2_numa_mem_bw.png" alt="Node2 NUMA memory bandwidth" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_single_node_torchrun_train_step.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_single_node_torchrun_train_step.png" alt="Single-node torchrun train-step curve" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_torchrun_train_step.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_torchrun_train_step.png" alt="Multinode torchrun train-step curve" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_tpot_vs_concurrency.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_tpot_vs_concurrency.png" alt="Single-node vLLM TPOT vs concurrency" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_vllm_serve_tpot_vs_concurrency.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_vllm_serve_tpot_vs_concurrency.png" alt="Multinode vLLM TPOT vs concurrency" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_c2c_memcpy_bw.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_c2c_memcpy_bw.png" alt="Node1 C2C memcpy bandwidth" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_c2c_memcpy_lat.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_c2c_memcpy_lat.png" alt="Node1 C2C memcpy latency" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_cluster_perf_grouped_gemm_tflops.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_cluster_perf_grouped_gemm_tflops.png" alt="Node1 grouped GEMM TFLOPS" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node2_cluster_perf_grouped_gemm_tflops.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node2_cluster_perf_grouped_gemm_tflops.png" alt="Node2 grouped GEMM TFLOPS" width="920"/></a></p>

Data: [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nccl.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nccl.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_numa_mem_bw.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_numa_mem_bw.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_numa_mem_bw.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_numa_mem_bw.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_single_node_torchrun_train_step.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_single_node_torchrun_train_step.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_torchrun_train_step.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_torchrun_train_step.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.csv), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_c2c_memcpy.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_c2c_memcpy.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_cluster_perf_grouped_gemm_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_cluster_perf_grouped_gemm_summary.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_cluster_perf_grouped_gemm_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_cluster_perf_grouped_gemm_summary.json)

## NVLink/NVSwitch Topology Snapshot
| Node | GPU count | NVLink pair count | Link class |
| --- | ---: | ---: | --- |
| node1 | `4` | `6/6` | `NV18` full mesh |
| node2 | `4` | `6/6` | `NV18` full mesh |

<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_meta_nvlink_topology.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_meta_nvlink_topology.png" alt="Node1 NVLink topology" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node2_meta_nvlink_topology.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node2_meta_nvlink_topology.png" alt="Node2 NVLink topology" width="920"/></a></p>

Data: [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_meta_nvlink_topology.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_meta_nvlink_topology.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_meta_nvlink_topology.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_meta_nvlink_topology.json)

## Dedicated nvbandwidth Snapshot
| Metric | node1 | node2 |
| --- | ---: | ---: |
| Status | `ok` | `ok` |
| Requested runtime | `host` | `host` |
| Effective runtime | `host` | `host` |
| SUM metric count | `43` | `43` |
| Peak SUM metric (`peak_sum_gbps`) | `20765.46` | `20766.81` |
| D2D bidir read CE total (GB/s) | `18323.23` | `18322.75` |
| D2D bidir write CE total (GB/s) | `18498.67` | `18499.85` |

Interpretation: nvbandwidth collection is now first-class in canonical run. Host runtime requires explicit compat-lib injection to resolve prior PTX incompatibility.

<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth_sums.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth_sums.png" alt="Node1 nvbandwidth sums" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth_sums.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth_sums.png" alt="Node2 nvbandwidth sums" width="920"/></a></p>

Data: [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json)

## GB200 Extensions (Enabled in Canonical Run)
| Extension | Key result | Evidence |
| --- | --- | --- |
| All-reduce stability | Mean bus BW `805.661 GB/s`; CV `2.189%`; jitter assessment `moderate_jitter` | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_stability.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_stability.json) |
| All-reduce latency composition | One-large vs many-small bandwidth ratio `6.817x` (`828.595` vs `121.550 GB/s`) | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_latency_comp.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_latency_comp.json) |
| Control-plane collective overhead | `all_reduce_tensor` fastest mean (`0.220 ms`); `all_gather_object` is `4.603x` slower than `all_gather_tensor` | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allgather_control_plane.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allgather_control_plane.json) |
| NCCL algo comparison | `auto=840.59`, `NVLS=839.70`, `Ring=698.42`, `Tree=546.54 GB/s` | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_algo_comparison.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_algo_comparison.json) |
| C2C memcpy | Pinned H2D/D2H peaks `125.830/124.536 Gbps`; 4-byte pinned latency `2.039/1.839 us` | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_c2c_memcpy.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_c2c_memcpy.json) |
| Train-step scaling | Single-node `103,170.572 tok/s` to multi-node `210,111.128 tok/s` (`2.036x`) | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_single_node_torchrun_train_step.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_single_node_torchrun_train_step.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_torchrun_train_step.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_torchrun_train_step.json) |
| FP4 skew guard | `pass`; max pairwise median gap `1.316%`; attestation consistency `pass` | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_fp4_smoke_skew_guard.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_fp4_smoke_skew_guard.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_fp4_attestation_consistency.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_fp4_attestation_consistency.json) |
| MAMF straggler check | Across 8 GPUs: `1550.30 -> 1730.47 TFLOPS` (`10.41%` gap-to-best, `11.62%` min-to-max spread) | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_gpu0_mamf_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_gpu0_mamf_summary.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_gpu3_mamf_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_gpu3_mamf_summary.json) |

<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_stability.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_stability.png" alt="All-reduce stability" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_latency_comp.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_latency_comp.png" alt="All-reduce latency composition" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_allgather_control_plane.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_allgather_control_plane.png" alt="Control-plane collective overhead" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_algo_comparison.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_algo_comparison.png" alt="NCCL algorithm comparison" width="920"/></a></p>
<p><a href="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_mamf_straggler.png"><img src="docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_mamf_straggler.png" alt="MAMF straggler spread" width="920"/></a></p>

Data: [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_stability.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_stability.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_latency_comp.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allreduce_latency_comp.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allgather_control_plane.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_allgather_control_plane.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_algo_comparison.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_algo_comparison.json)

## Required Issues (Explicit)
| Required issue (verbatim) | Current status | Evidence |
| --- | --- | --- |
| Missing node2 fio artifact in canonical package (node2_fio.json absent). | Resolved. Artifact is present in canonical run. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_fio.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_fio.json) |
| No multinode vLLM artifact in canonical package. | Resolved. JSON/CSV/JSONL and clock-lock artifacts are present. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.csv) |
| No nvbandwidth bundle in canonical package. | Resolved. Node1+node2 nvbandwidth artifacts are present and status is `ok`. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json) |
| Health suite had GDR requested, but effective GDR was false due non-CUDA IB local checks. | Resolved for canonical run (`requested=true`, `effective_enabled=true`). Additional context: unsupported mem-type probes (`cuda_mem_type=1`) were root-caused and canonical coverage is now fail-fast validated for supported mem types. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json), [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_suite/health_suite_extended.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_suite/health_suite_extended.log) |
| Tail latency knee is severe at high concurrency (throughput up, TTFT/p99 TTFT much worse). | Confirmed and still open risk. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv), [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_ttft_vs_concurrency.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_ttft_vs_concurrency.png) |

## Root Cause + Fix Mapping
| Issue | Root cause | Fix shipped | Canonical result |
| --- | --- | --- | --- |
| Missing `node2_fio.json` | All-node storage capture was not hard-gated in earlier flow. | Required all-node fio collection and validation in suite. | `node2_fio.json` present and linked in manifest. |
| Missing multinode vLLM artifact | Earlier multinode run lifecycle could end with worker non-zero (`137`) during teardown. | Worker teardown normalized to intentional-stop semantics only when leader completed and output exists; strict return-code validation retained. | Multinode vLLM status `ok`, rc clean, artifacts complete. |
| Missing nvbandwidth bundle | Host nvbandwidth failed with PTX/user-mode compatibility mismatch. | Added host compat-lib extraction/injection path; host runtime now succeeds without runtime fallback switching. | Node1+node2 nvbandwidth bundles present, `effective_runtime=host`. |
| GDR requested but ineffective | Earlier prerequisite checks and unsupported mode handling created false-negative outcomes. | Strict preflight + explicit mem-type probe validation before running health checks; unsupported modes fail early. | Canonical health suite reports GDR effective true. |
| Severe latency knee | Queueing/saturation at high concurrency. | No data-path bug to patch; kept as explicit risk with operating envelope guidance. | Risk remains and is measured in canonical sweep. |

## Report Completeness Delta (vs prior condensed revision)
| Area | Prior condensed state | Restored now |
| --- | --- | --- |
| Canonical run alignment | Pointed to `2026-02-09_fresh_full_suite_e2e_fixed` with remediation context | Fully synced to `2026-02-10_full_suite_e2e_wire_qf_mon` |
| Benchmark/report depth | Basic A/B sections only | Restored expanded merged weird/normal section (`Baseline vs Weird Log` + `Deep-Dive Findings`), NVLink snapshot, dedicated nvbandwidth, GB200 extensions, completeness delta |
| Visual coverage | Smaller figure set | Canonical report now references `33` canonical figures |
| Historical incident appendix | Older report included incident-era (non-canonical) artifact links | Superseded incident artifacts were intentionally cleaned; operator lessons were retained and rewritten against canonical evidence in the merged weird/normal section |
| Required issue handling | Present but tied to older canonical package | Revalidated issue ledger against green canonical artifacts |
| Suite status clarity | Mixed with remediation notes | Clean `53/53` green suite-step narrative with explicit evidence |

## Gaps, Risks, and Smell Checks
| Severity | Finding | Why it matters | Evidence |
| --- | --- | --- | --- |
| High | Severe TTFT tail knee at high concurrency (`c=512`) and degraded operator checks (tooling gaps) on both nodes. | Throughput-optimized mode is not interactive-latency safe, and missing operational tools increase MTTR during incidents. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json) |
| Medium | GDR coverage is valid but limited to supported mem-type path in canonical run (`mem_types=0` + dmabuf variant). | Requested unsupported mem-types should remain explicit preflight failures, not silent downgrades. | [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_suite/health_suite_extended.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_suite/health_suite_extended.log) |
| Medium | MAMF variability across 8 GPUs is non-trivial (`10.41%` gap-to-best; `11.62%` min-to-max spread). | Potential straggler/placement sensitivity should be trended over time. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_gpu0_mamf_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_gpu0_mamf_summary.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_gpu3_mamf_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_gpu3_mamf_summary.json) |
| Low | OOB TCP is much slower than IB. | Control-plane path is not data-plane substitute; launcher pinning remains important. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_iperf3_oob_tcp.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_iperf3_oob_tcp.json) |

## Implications for Small AI Teams
| Area | Practical implication |
| --- | --- |
| Cluster onboarding | You can get to first credible multinode signals quickly if preflight + interface pinning are codified. |
| Serving policy | Define two profiles: latency-safe (`<=256`) and overflow throughput (`512`) with explicit SLA caveats. |
| Run hygiene | Keep canonical-only report linkage and aggressively remove superseded intermediate runs to avoid stale evidence drift. |
| Operator readiness | Treat `quick_friction` + `monitoring_expectations` as required gate outputs, and close the missing-tool gaps before major production launch milestones. |
| Reliability culture | Enforce fail-fast for unsupported requested modes (especially GDR mem-type matrices) instead of warning-only continuation. |
| Capacity planning | Track straggler spread (MAMF + GEMM parity) as an ongoing fleet-health metric, not a one-off. |

## Stakeholder Recommendations (Prioritized)
| Priority | Recommendation |
| --- | --- |
| `P0` | Keep `run_cluster_eval_suite.sh` canonical runs green-only: if required checks fail, rerun only after root-cause fix, not by accepting remediation context as canonical. |
| `P0` | Keep vLLM serving policy split (`latency mode` vs `throughput mode`) and publish hard concurrency guardrails. |
| `P0` | Keep `quick_friction` and `monitoring_expectations` as required report sections and required artifact rows in reproducibility packages for every canonical run. |
| `P1` | Keep strict GDR mem-type preflight probes and fail early on unsupported requested modes. |
| `P1` | Keep host nvbandwidth compat-lib prep in the default path for this cluster profile until host user-mode stack is harmonized. |
| `P1` | Close currently failing quick-friction checks by installing/authorizing `huggingface-cli`, `whois`, and `speedtest` on both nodes. |
| `P1` | Close currently failing monitoring checks by providing `kubectl` context and safe privileged `dmesg` ingestion path (or explicit approved alternative source). |
| `P2` | Add repeated-run trend snapshots for MAMF spread, allreduce stability CV, and serving tail-latency p99. |

## Repro Steps
Canonical full-suite command:

```bash
cd code/cluster

scripts/run_cluster_eval_suite.sh \
  --run-id 2026-02-10_full_suite_e2e_wire_qf_mon \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --oob-if enP22p3s0f3 \
  --socket-ifname enP22p3s0f3 \
  --nccl-ib-hca mlx5_0,mlx5_1,mlx5_4,mlx5_5 \
  --health-suite extended \
  --health-gdr \
  --health-gdr-gpu 0 \
  --health-gdr-mem-types 0 \
  --health-gdr-use-dmabuf \
  --run-vllm-multinode \
  --run-nvbandwidth \
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

## Reproducibility Package
| Artifact class | Canonical artifact |
| --- | --- |
| Manifest | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_manifest.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_manifest.json) |
| Suite steps | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json) |
| Discovery/meta | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_meta.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_meta.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_meta.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_meta.json) |
| Health summary | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json) |
| Single-node serving | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv) |
| Multinode serving | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.json) |
| nvbandwidth | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json) |
| fio all nodes | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_fio.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_fio.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_fio.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_fio.json) |
| Required reliability-gate package | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_torchrun_connectivity_probe.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_torchrun_connectivity_probe.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_hang_triage_readiness.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_hang_triage_readiness.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_hang_triage_readiness.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_hang_triage_readiness.json) |
| Quick friction package | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_quick_friction.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_quick_friction.json) |
| Monitoring expectations package | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_monitoring_expectations.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_monitoring_expectations.json) |
| Artifact inventory lists | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_artifacts_summary.txt](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_artifacts_summary.txt), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_artifacts_structured.txt](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_artifacts_structured.txt), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_artifacts_raw.txt](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_artifacts_raw.txt), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_artifacts_figures.txt](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_artifacts_figures.txt) |
| Topology visuals | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_meta_nvlink_topology.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_meta_nvlink_topology.png), [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node2_meta_nvlink_topology.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node2_meta_nvlink_topology.png) |

## Activity Log
| Date (UTC) | Action |
| --- | --- |
| 2026-02-09 | Root-caused and fixed multinode vLLM worker teardown/rc handling so canonical summary is clean when intentional shutdown occurs post-success. |
| 2026-02-09 | Root-caused and fixed nvbandwidth host PTX incompatibility by injecting CUDA compat libs into host runtime path; removed runtime fallback dependence in canonical path. |
| 2026-02-09 | Re-synced scripts to both nodes and verified nvbandwidth all-nodes host path with compat libs. |
| 2026-02-09 to 2026-02-10 | Ran full canonical suite end-to-end with fresh run id `2026-02-10_full_suite_e2e_wire_qf_mon`; all suite steps green. |
| 2026-02-10 | Regenerated stakeholder report to canonical run and restored richer report sections/visual coverage. |
| 2026-02-10 | Performed targeted cleanup of superseded intermediate artifacts while preserving canonical run artifacts. |
| 2026-02-10 | Added case-study anti-regression guardrails: AGENTS contract updates, template retention gates, and report validator script with zero-failure pass. |
| 2026-02-10 | Canonical run executed required reliability gates and newly wired `quick_friction` + `monitoring_expectations` checks, producing explicit degraded diagnostics for missing tools/permissions without masking benchmark validity. |

## Appendix (Coverage vs Case-Study Goals)
| Case-study requirement | Coverage status | Where addressed | Evidence |
| --- | --- | --- | --- |
| Tell us a story about this cluster | Covered | `Cluster Story (First Contact)` | [#cluster-story-first-contact](#cluster-story-first-contact), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json) |
| What is weird, new, or interesting | Covered | `Weird / New / Interesting (with Normal Baseline)` with `Baseline vs Weird Log` + `Deep-Dive Findings` subsections | [#weird--new--interesting-with-normal-baseline](#weird--new--interesting-with-normal-baseline), [#baseline-vs-weird-log](#baseline-vs-weird-log), [#deep-dive-findings](#deep-dive-findings) |
| Describe technical experience with 1-2 relevant benchmarks for a small AI team | Covered | `Benchmark A` (networking) + `Benchmark B` (inference) | [#benchmark-a-networking-story](#benchmark-a-networking-story), [#benchmark-b-inference-story](#benchmark-b-inference-story) |
| Make benchmarking repeatable and save structured outputs | Covered | `Repro Steps` + `Reproducibility Package` | [#repro-steps](#repro-steps), [#reproducibility-package](#reproducibility-package), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_manifest.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_manifest.json) |
| Include visualizations to tell the story | Covered | benchmark arcs + supporting deep-dive visuals | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png), [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_2nodes_nccl_bw_vs_msg.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_2nodes_nccl_bw_vs_msg.png), [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_ttft_vs_concurrency.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_ttft_vs_concurrency.png) |
| Provide insights from provider experience not obvious elsewhere | Covered | `Gaps, Risks, and Smell Checks` + `Implications for Small AI Teams` | [#gaps-risks-and-smell-checks](#gaps-risks-and-smell-checks), [#implications-for-small-ai-teams](#implications-for-small-ai-teams) |
| Keep report table-forward with high-value sections preserved | Covered | enforced by template + validator + current report section layout | [cluster/docs/field-report-template.md](docs/field-report-template.md), [cluster/scripts/validate_field_report_requirements.sh](scripts/validate_field_report_requirements.sh), [#table-of-contents](#table-of-contents) |
| Preserve new operator checks in stakeholder artifacts | Covered | `Operator Friction + Monitoring Expectations (New Checks)` + `Reproducibility Package` | [#operator-friction--monitoring-expectations-new-checks](#operator-friction--monitoring-expectations-new-checks), [#reproducibility-package](#reproducibility-package), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json) |
