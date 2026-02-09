# Cluster Case Study Field Notes (Synchronized)

Last updated: 2026-02-09 (post-remediation sync).

## Table of Contents
1. [Scope](#scope)
2. [Synchronization Status](#synchronization-status)
3. [Required Issue Ledger](#required-issue-ledger)
4. [Root Cause + Fix Mapping](#root-cause--fix-mapping)
5. [Evidence Matrix](#evidence-matrix)
6. [Smell Checks](#smell-checks)
7. [Repro Entry Point](#repro-entry-point)

## Scope
| Item | Value |
| --- | --- |
| In-scope nodes | `node1`, `node2` (4x GB200 per node, 8 GPUs total) |
| Excluded nodes | none |
| Canonical run | `2026-02-09_fresh_full_suite_e2e_fixed` |
| Canonical manifest | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_manifest.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_manifest.json) |
| Canonical suite steps (first full pass) | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_suite_steps.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_suite_steps.json) |
| Canonical remediation status | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_remediation_status.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_remediation_status.json) |

## Synchronization Status
| Surface | Canonical run aligned | Notes |
| --- | --- | --- |
| `field-report.md` | yes | Rewritten with remediated canonical metrics/artifacts |
| `field-report-notes.md` | yes | Mirrors same issue states and evidence links |
| Evidence links | yes | Links point only to preserved canonical package |
| Superseded artifacts | cleaned | Non-canonical 2026 intermediates removed from `results/structured`, `results/raw`, `docs/figures` |

## Required Issue Ledger
| Required issue | Status now | Evidence |
| --- | --- | --- |
| Missing node2 fio artifact in canonical package (`node2_fio.json` absent). | Resolved | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node2_fio.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node2_fio.json) |
| No multinode vLLM artifact in canonical package. | Resolved | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_multinode_serve.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_multinode_serve.json)<br/>[results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_multinode_serve.csv](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_multinode_serve.csv) |
| No nvbandwidth bundle in canonical package. | Resolved | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_nvbandwidth.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_nvbandwidth.json)<br/>[results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node2_nvbandwidth.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node2_nvbandwidth.json) |
| Health suite had GDR requested, but effective GDR was false due non-CUDA IB local checks. | Resolved | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_health_suite_extended_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_health_suite_extended_node1node2_cluster_health_suite_summary.json) |
| Tail latency knee is severe at high concurrency (throughput up, TTFT/p99 TTFT much worse). | Confirmed ongoing risk | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_serve_sweep.csv](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_serve_sweep.csv)<br/>[docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_serve_ttft_vs_concurrency.png](docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_serve_ttft_vs_concurrency.png) |

## Root Cause + Fix Mapping
| Required issue | Why it happened | Fix applied | Current state |
| --- | --- | --- | --- |
| Missing `node2_fio.json` | All-node storage evidence was not hard-enforced in initial package. | Enforced all-node fio collection and required-artifact validation. | Canonical package now includes node2 fio. |
| Missing multinode vLLM artifact | Multinode serve outputs were not guaranteed in first-pass package. | Enforced required multinode vLLM artifacts + lock files and reran multinode serve. | Canonical package now includes multinode vLLM evidence. |
| Missing nvbandwidth bundle | Host nvbandwidth execution failed due PTX/toolchain mismatch. | Added host->container fallback metadata and switched default image to pullable parity image. | Canonical package now includes node1/node2 nvbandwidth artifacts. |
| GDR requested but effective false | False-negative prereq path in health-suite local GDR checks under strict shell behavior. | Fixed check path and reran health suite with GDR enabled. | Canonical health summary now reports effective GDR true. |
| Severe high-concurrency tail-latency knee | Saturation behavior, not artifact-loss bug. | Kept as measured risk and documented policy implication. | Still present; explicitly flagged. |

## Evidence Matrix
| Claim | Data Evidence | Visualization Evidence | Status |
| --- | --- | --- | --- |
| Networking fabric is healthy in canonical run (`all_reduce` peak `839.27 GB/s`). | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_health_suite_extended_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_health_suite_extended_node1node2_cluster_health_suite_summary.json) | [docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_2nodes_nccl_bw_vs_msg.png](docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_2nodes_nccl_bw_vs_msg.png) | Backed |
| vLLM single-node knee is severe at top throughput (`c=512`, mean TTFT `5595.330 ms`, p99 TTFT `12085.465 ms`). | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_serve_sweep.csv](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_serve_sweep.csv) | [docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_serve_ttft_vs_concurrency.png](docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_serve_ttft_vs_concurrency.png) | Backed |
| Multinode vLLM canary is captured in canonical package (`c=64`, status `ok`). | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_multinode_serve.csv](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_multinode_serve.csv) | [docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_node1_multinode_vllm_serve_total_tok_s_vs_concurrency.png](docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_node1_multinode_vllm_serve_total_tok_s_vs_concurrency.png) | Backed |
| Storage parity now includes both nodes. | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_fio.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_fio.json)<br/>[results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node2_fio.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node2_fio.json) | [docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_node1_fio.png](docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_node1_fio.png)<br/>[docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_node2_fio.png](docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_node2_fio.png) | Backed |
| NVBandwidth bundle is captured on both nodes with runtime fallback metadata. | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_nvbandwidth.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_nvbandwidth.json)<br/>[results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node2_nvbandwidth.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node2_nvbandwidth.json) | [docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_node1_nvbandwidth_sums.png](docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_node1_nvbandwidth_sums.png)<br/>[docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_node2_nvbandwidth_sums.png](docs/figures/2026-02-09_fresh_full_suite_e2e_fixed_node2_nvbandwidth_sums.png) | Backed |
| GDR effective flag is true in canonical health summary. | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_health_suite_extended_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_health_suite_extended_node1node2_cluster_health_suite_summary.json) | n/a | Backed |

## Smell Checks
| Severity | Smell | Why it matters | Evidence |
| --- | --- | --- | --- |
| High | `suite_steps` still carries 3 first-pass failures despite remediated artifacts. | Step ledger and final canonical state can drift unless remediation is explicitly tracked. | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_suite_steps.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_suite_steps.json)<br/>[results/structured/2026-02-09_fresh_full_suite_e2e_fixed_remediation_status.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_remediation_status.json) |
| Medium | GDR mem-type matrix is partial (`ib_gdr` records `mem0` + `mem0_dmabuf`; mem1 subtests failed as warnings). | Effective GDR is true, but mem-type coverage is narrower than requested matrix. | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_health_suite_extended_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_health_suite_extended_node1node2_cluster_health_suite_summary.json)<br/>[results/raw/2026-02-09_fresh_full_suite_e2e_fixed_health_suite_extended_node1node2_cluster_health_suite.log](results/raw/2026-02-09_fresh_full_suite_e2e_fixed_health_suite_extended_node1node2_cluster_health_suite.log) |
| Medium | Tail-latency knee remains severe at high concurrency. | Throughput-optimal operating point is not latency-safe for interactive workloads. | [results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_serve_sweep.csv](results/structured/2026-02-09_fresh_full_suite_e2e_fixed_node1_vllm_serve_sweep.csv) |

## Repro Entry Point
Use the commands in [field-report.md](field-report.md#repro-steps).
