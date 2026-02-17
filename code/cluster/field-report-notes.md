# Cluster Case Study Field Notes (Synchronized)

Last updated: 2026-02-10. Canonical run: `2026-02-10_full_suite_e2e_wire_qf_mon`.

## Table of Contents
1. [Scope](#scope)
2. [Required Reliability Gates (Canonical Run)](#required-reliability-gates-canonical-run)
3. [Operator Friction + Monitoring Expectations (New Checks)](#operator-friction--monitoring-expectations-new-checks)
4. [Synchronization Status](#synchronization-status)
5. [Required Issue Ledger](#required-issue-ledger)
6. [Root Cause + Fix Mapping](#root-cause--fix-mapping)
7. [Evidence Matrix](#evidence-matrix)
8. [Coherence + Smell Checks](#coherence--smell-checks)
9. [Repro Entry Point](#repro-entry-point)

## Scope
| Item | Value |
| --- | --- |
| Hosts | `node1,node2` |
| GPUs | `4` per host |
| Excluded hosts | none |
| Canonical run | `2026-02-10_full_suite_e2e_wire_qf_mon` |
| Manifest | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_manifest.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_manifest.json) |
| Suite steps | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json) |
| Required-gate execution | Included in canonical run (`2026-02-10_full_suite_e2e_wire_qf_mon`) |
| Suite status | `53/53` green steps |

## Required Reliability Gates (Canonical Run)
| Gate | Status | Evidence |
| --- | --- | --- |
| Hang-triage readiness (`py-spy` + `strace`) | `ok` on node1+node2 | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_hang_triage_readiness.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_hang_triage_readiness.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_hang_triage_readiness.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_hang_triage_readiness.json) |
| Torchrun connectivity probe | `ok` (`world_size=8`, barrier mean `0.137 ms`, `p95=0.230 ms`) | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_torchrun_connectivity_probe.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_torchrun_connectivity_probe.json) |
| NCCL env sensitivity sweep | `ok` (`failure_count=0`), best profile `baseline_auto` (`1.000000x` vs baseline) | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.json), [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.png) |
| Canonical manifest | present | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_manifest.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_manifest.json) |

## Operator Friction + Monitoring Expectations (New Checks)
| Check | node1 | node2 | Evidence |
| --- | --- | --- | --- |
| quick_friction | `degraded` (`4/7` pass) | `degraded` (`4/7` pass) | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_quick_friction.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_quick_friction.json) |
| monitoring_expectations | `degraded` (`gpu_telemetry=ok`, `control_plane/system_signals=error`) | `degraded` (`gpu_telemetry=ok`, `control_plane/system_signals=error`) | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_monitoring_expectations.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_monitoring_expectations.json) |
| raw logs | present | present | [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.log), [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.log), [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node2_quick_friction.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node2_quick_friction.log), [results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node2_monitoring_expectations.log](results/raw/2026-02-10_full_suite_e2e_wire_qf_mon_node2_monitoring_expectations.log) |

## Synchronization Status
| Surface | Status | Notes |
| --- | --- | --- |
| `field-report.md` | synced | All links now point to `2026-02-10_full_suite_e2e_wire_qf_mon`. |
| `field-report-notes.md` | synced | Same canonical run, issue ledger, and metrics framing. |
| Required-gate references | synced | Report + notes both include canonical gate artifacts and env-sensitivity visual. |
| Operator checks | synced | Report + notes include dedicated quick-friction/monitoring sections with node-level artifacts and raw logs. |
| Canonical suite status | green | `53/53` steps green; required artifact validation passed. |
| Visual coverage | restored | Primary report now references all `33` canonical figures (including per-node deep-dive bundles). |
| Weird/normal depth | restored | Report now uses one merged section (`Weird / New / Interesting (with Normal Baseline)`) with two subsections (`Baseline vs Weird Log`, `Deep-Dive Findings`) so operator reality and weirdness are unified without duplication. |
| Historical incident linkage | normalized | Prior incident-era links in older report revisions were superseded during cleanup; current report retains those lessons with canonical-run-backed evidence only. |
| Case-study requirement mapping | restored | Appendix now maps each explicit case-study prompt requirement to report sections + evidence links. |
| Guardrails against section loss | enabled | Template + AGENTS + validator script now enforce required sections and table-forward high-value coverage. |

## Required Issue Ledger
| Required issue (verbatim) | Status now | Evidence |
| --- | --- | --- |
| Missing node2 fio artifact in canonical package (node2_fio.json absent). | Resolved | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_fio.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_fio.json) |
| No multinode vLLM artifact in canonical package. | Resolved | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.json) |
| No nvbandwidth bundle in canonical package. | Resolved | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json) |
| Health suite had GDR requested, but effective GDR was false due non-CUDA IB local checks. | Resolved in canonical run (`requested=true`, `effective_enabled=true`) | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json) |
| Tail latency knee is severe at high concurrency (throughput up, TTFT/p99 TTFT much worse). | Confirmed ongoing risk | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv), [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_ttft_vs_concurrency.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_ttft_vs_concurrency.png) |

## Root Cause + Fix Mapping
| Issue | Root cause | Fix |
| --- | --- | --- |
| Missing `node2_fio.json` | All-node storage capture was not hard-gated in earlier run packaging. | All-node fio collection + required artifact validation in suite flow. |
| Missing multinode vLLM artifacts | Worker lifecycle teardown could produce non-clean rc in earlier multinode run packaging. | Explicit worker stop semantics + rc normalization only for intentional shutdown after success. |
| Missing nvbandwidth bundle | Host nvbandwidth failed due PTX/user-mode mismatch. | Host compat-lib extraction/injection path; canonical host runtime now succeeds. |
| GDR requested but ineffective | Prereq/mode validation and unsupported mode handling were not strict enough in older path. | Strict prereq + mem-type probe validation before run; unsupported requests fail early. |
| Tail latency knee | Saturation behavior, not missing artifact bug. | Keep explicit risk framing and serving policy guardrails. |
| quick_friction degraded state | `huggingface-cli`, `whois`, and `speedtest` binaries are absent on both nodes. | Keep check required and close tool gaps before production launch milestones. |
| monitoring_expectations degraded state | `kubectl` context/tooling absent and `dmesg` access denied on both nodes. | Keep check required and provision control-plane context + approved kernel log ingestion path. |

## Evidence Matrix
| Claim | Data evidence | Visual evidence | Verdict |
| --- | --- | --- | --- |
| Canonical suite is fully green. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_suite_steps.json) | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png) | Backed |
| Networking story is strong and stable. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_health_suite_extended_node1node2_cluster_health_suite_summary.json) | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_2nodes_nccl_bw_vs_msg.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_2nodes_nccl_bw_vs_msg.png) | Backed |
| Inference knee remains severe at high concurrency. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_sweep.csv) | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_ttft_vs_concurrency.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_serve_ttft_vs_concurrency.png) | Backed |
| Multinode vLLM evidence is now complete. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_vllm_multinode_serve.json) | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_vllm_serve_total_tok_s_vs_concurrency.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_multinode_vllm_serve_total_tok_s_vs_concurrency.png) | Backed |
| nvbandwidth bundle exists and is coherent across nodes. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_nvbandwidth.json) | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth_sums.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_node1_nvbandwidth_sums.png) | Backed |
| Merged weird/normal section retains historical intent with canonical evidence. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_preflight_services.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_preflight_services.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node_parity_summary.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node_parity_summary.json) | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png), [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_mamf_straggler.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_mamf_straggler.png) | Backed |
| New required reliability gates are executable and semantically green on in-scope hosts. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_manifest.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_manifest.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_torchrun_connectivity_probe.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_torchrun_connectivity_probe.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.json) | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_nccl_env_sensitivity.png) | Backed |
| Quick friction and monitoring expectation checks are now first-class required evidence. | [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_quick_friction.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_quick_friction.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_quick_friction.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node1_monitoring_expectations.json), [results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_monitoring_expectations.json](results/structured/2026-02-10_full_suite_e2e_wire_qf_mon_node2_monitoring_expectations.json) | [docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png](docs/figures/2026-02-10_full_suite_e2e_wire_qf_mon_cluster_story_dashboard.png) | Backed |

## Coherence + Smell Checks
| Severity | Check | Outcome |
| --- | --- | --- |
| High | Any canonical link still points to `2026-02-09_fresh_full_suite_e2e_fixed` or `...green_strict`? | No |
| High | Any required issue unresolved in canonical package? | No for artifact presence; latency knee remains intentionally open risk |
| High | Any required-gate artifact missing for `2026-02-10_full_suite_e2e_wire_qf_mon`? | No |
| High | Any quick-friction artifact missing for canonical run? | No |
| High | Any monitoring-expectations artifact missing for canonical run? | No |
| Medium | GDR requested/effective mismatch in canonical summary? | No (`true/true`) |
| Medium | Root-cause fix replaced by fallback semantics in canonical path? | No. Canonical nvbandwidth path is host runtime with explicit compat libs |
| Medium | Any required-gate status is non-`ok`? | No (`hang_triage`: node1/node2 `ok`; `connectivity_probe`: `ok`; `nccl_env_sensitivity`: `ok`) |
| Medium | Are new operator checks degraded? | Yes by design in this run; diagnostics are explicit (`kubectl`/`dmesg` + external-tool gaps) and tracked as actionable follow-up. |

## Repro Entry Point
Use the canonical command in [field-report.md](field-report.md), section `Repro Steps`.
