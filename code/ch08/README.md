# Chapter 8 - Occupancy & Pipeline Tuning

## Summary
Concentrates on resource balancing: adjust block sizes, registers, and shared memory to keep SMs full while hiding TMEM latency via double buffering, loop unrolling, and async pipelines.

## Problem
Chapter 8 is where pipeline and occupancy theory has to survive contact with a real kernel. The useful question is not "is occupancy important?" but "which tuning changes actually improve the measured path once register pressure, staging, and pipeline depth all interact?"

## Baseline Path
- conservative block sizes and staging behavior
- less overlap between memory movement and compute
- easier to reason about, but often leaves the SM underfilled or the pipeline underutilized

## Optimized Path
- occupancy-aware launch and block-shape tuning
- more aggressive staging or threshold/TMA pipeline behavior where it helps
- measured through the same harness contract as the rest of the repo, so the gains are not one-off microbench stories

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `threshold` | `3.568 ms` | `0.295 ms` | `12.11x` | better threshold pipeline and staging behavior |
| `occupancy_tuning` | `0.092 ms` | `0.014 ms` | `6.80x` | launch/block tuning for better resident work |
| `tcgen05_tiling_vs_cublas` | `0.713 ms` | `0.154 ms` | `4.63x` | better tiling schedule against the baseline path |

This chapter is good for showing that occupancy improvements are usually not isolated wins. They tend to land together with better staging and fewer pipeline stalls.

## Profiler Evidence
Use deep-dive harness runs when you want to see whether the improvement came from occupancy, staging overlap, or less pipeline idle time:

```bash
python -m cli.aisp bench run --targets ch08:threshold --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch08:occupancy_tuning --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch08:tcgen05_tiling_vs_cublas --profile deep_dive --single-gpu
```

Those targets give you three useful slices:
- `threshold`: threshold/pipeline behavior
- `occupancy_tuning`: block-shape and resident-work tuning
- `tcgen05_tiling_vs_cublas`: tiling and tensor-core schedule quality

## Repro Commands
```bash
python -m ch08.compare
python -m cli.aisp bench list-targets --chapter ch08
python -m cli.aisp bench run --targets ch08 --profile minimal
python -m cli.aisp bench run --targets ch08:threshold --profile deep_dive --single-gpu
```

## Learning Goals
- Tune occupancy explicitly and observe how register counts limit resident CTAs.
- Apply double buffering and async staging to overlap DRAM fetch with compute.
- Use tiling, loop unrolling, and AI-specific thresholds to control latency vs throughput.
- Measure how pipelined schedules change SM/TMEM utilization using the shared harness.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_occupancy_tuning.py`, `optimized_occupancy_tuning.py`, `occupancy_tuning_tool.py`, `occupancy_api_example.cu`, `occupancy_tuning.cu` | Occupancy studies that tune CTA shapes, register caps, and API-computed limits (plus a sweep tool for quick preset exploration). |
| `baseline_ai_optimization.py`, `optimized_ai_optimization.py`, `ai_optimization_kernels.cu`, `independent_ops.cu` | AI kernel scheduling samples that stage independent ops to highlight pipeline and occupancy tradeoffs. |
| `baseline_hbm_cuda.cu`, `baseline_hbm_cuda.py`, `baseline_hbm.py`, `optimized_hbm.py`, `optimized_hbm_cuda_vectorized.cu`, `optimized_hbm_cuda_vectorized.py`, `hbm_kernels.cu` | HBM streaming workloads that compare scalar, vectorized, and asynchronous fetch patterns. |
| `baseline_loop_unrolling.cu`, `baseline_loop_unrolling.py`, `optimized_loop_unrolling.cu`, `optimized_loop_unrolling.py`, `loop_unrolling_kernels.cu` | Loop unrolling case studies targeting various ILP regimes. |
| `baseline_threshold.py`, `baseline_thresholdtma.py`, `optimized_threshold.py`, `optimized_thresholdtma.py`, `threshold_kernels.cu`, `threshold_tma_benchmark_base.py` | Threshold operators implemented with scalar, vectorized, and TMA-backed pipelines. |
| `baseline_tiling.py`, `baseline_tiling_tcgen05.py`, `optimized_tiling.py`, `optimized_tiling_tcgen05.py`, `tiling_kernels.cu`, `tiling_extension_tcgen05.py` | Tile schedulers for tcgen05 matmuls, including safe fallbacks when tcgen05 isn't available. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json` | Harness entry, dependencies, and regression thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch08.compare
python -m cli.aisp bench list-targets --chapter ch08
python -m cli.aisp bench run --targets ch08 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- Nsight Compute traces for `optimized_thresholdtma.py` should show overlapping TMA loads with minimal idle cycles.
- `python -m cli.aisp tools occupancy-tuning` prints preset timings + speedups for the occupancy tuning microbenchmark.
- `python -m ch08.compare --examples threshold` confirms the TMA-backed kernels reducing latency vs scalar reference implementations.

## Notes
- `arch_config.py` exposes toggles for enabling/disabling tcgen05 lowering per GPU so the same scripts work on SM100 and SM121.
- `build/` caches CUDA object files per configuration; clean via `python cleanup.py --include-build` when adjusting toolchains.
