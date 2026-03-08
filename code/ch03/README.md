# Chapter 3 - System Tuning

## Summary
Captures the host-level changes-NUMA pinning, governor tweaks, container settings, and Kubernetes manifests-that keep GPU workloads fed before kernel-level optimization begins.

## Learning Goals
- Diagnose CPU and memory affinity issues that throttle GPU pipelines.
- Harden Docker and Kubernetes environments for sustained GPU throughput on shared clusters.
- Automate repeatable system tuning via shell scripts so lab machines stay consistent.
- Quantify how host-level fixes raise GEMM throughput and reduce launch latency.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_numa_unaware.py`, `optimized_numa_unaware.py`, `bind_numa_affinity.py`, `numa_topology_script.sh` | NUMA diagnostics and pinning helpers for binding data loaders, NCCL ranks, and GPU contexts to the right CPU sockets. |
| `baseline_docker.py`, `optimized_docker.py`, `docker_gpu_optimized.dockerfile`, `system_tuning.sh`, `gpu_setup_commands.sh` | Container configs plus host setup scripts that toggle persistence mode, huge pages, IRQ steering, and MIG visibility. |
| `baseline_kubernetes.py`, `optimized_kubernetes.py`, `kubernetes_mig_pod.yaml`, `kubernetes_topology_pod.yaml` | Kubernetes manifests demonstrating topology-aware scheduling and MIG partitioning for multi-tenant fleets. |
| `cpu_gpu_numa_optimizations.sh`, `system_tuning.sh`, `gpu_setup_commands.sh` | Workflow scripts for aligning CPU governors, cgroup limits, persistence mode, and driver settings with the benchmark harness. |
| `baseline_gemm.py`, `optimized_gemm.py`, `train.py` | Simple GEMM + training loops that surface the impact of system tuning changes in measurable FLOP/s. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json` | Harness entry, Python deps, and regression thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch03.compare
python -m cli.aisp bench list-targets --chapter ch03
python -m cli.aisp bench run --targets ch03 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m ch03.mig_mps_tool --device 0` reports the active MIG/MPS state before you change host-level scheduling policy.
- `python -m ch03.power_tuning_tool --power-limits 300,350 --iterations 5 --warmup 1` produces a short perf-per-watt sweep with the harness clock-lock path.
- `python -m ch03.compare` keeps the chapter baseline/optimized tuning pairs runnable through the shared harness.

## Notes
- `cpu_gpu_numa_optimizations.sh` is safe to rerun after every reboot; it re-applies irqbalance pinning and governor settings.
- Kubernetes manifests document the necessary annotations for NVLink/NVSwitch affinity without pointing to external repos.
