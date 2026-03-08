# Chapter 20 - End-to-End Case Studies

## Summary
Combines kernel, memory, pipeline, and inference optimizations into holistic case studies: take a baseline pipeline, apply staged improvements, and capture proof-of-benefit artifacts for every major subsystem.

## Problem
Chapter 20 is where isolated wins have to survive contact with the full stack. The useful question is not "did one optimization help in isolation?" but "what still matters after memory, pipeline, and inference optimizations are stacked together in one end-to-end workload?"

## Baseline Path
- sequential or minimally optimized end-to-end execution
- independent subsystems with little cross-stage coordination
- useful as a proof baseline, but usually leaves bandwidth and overlap on the table

## Optimized Path
- staged pipeline, memory, and KV-cache optimizations combined into one workload
- the same harness contract as every other chapter, so the end-to-end gains stay comparable to the lower-level chapters
- better for answering whether the optimizations compose cleanly instead of fighting each other

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `integrated_kv_cache` | `456.705 ms` | `67.381 ms` | `6.78x` | integrated KV-cache and overlap path |
| `pipeline_sequential` | `27.927 ms` | `1.683 ms` | `16.60x` | sequential pipeline replaced by coordinated staged execution |
| `multiple_unoptimized` | `0.616 ms` | `0.234 ms` | `2.63x` | stacked subsystem cleanup versus the intentionally rough composite baseline |

This chapter is the best place to check whether wins compose. A chapter 20 speedup is more meaningful than a microbench speedup when you want to know what survives in a real end-to-end path.

## Profiler Evidence
Use deep-dive harness runs when you want to see how the end-to-end gain breaks down by subsystem:

```bash
python -m cli.aisp bench run --targets ch20:integrated_kv_cache --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch20:pipeline_sequential --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch20:multiple_unoptimized --profile deep_dive --single-gpu
```

That is the right place to answer whether the gain came from overlap, memory movement, or simply removing one obvious bottleneck from the baseline.

## Repro Commands
```bash
python -m ch20.compare
python -m cli.aisp bench list-targets --chapter ch20
python -m cli.aisp bench run --targets ch20 --profile minimal
python -m cli.aisp bench run --targets ch20:pipeline_sequential --profile deep_dive --single-gpu
```

## Learning Goals
- Chain memory, pipeline, and KV-cache optimizations together to see cumulative impact.
- Generate automatic reports that compare baseline vs tuned end-to-end runs.
- Prototype new kernels via the AI kernel generator and slot them into the harness.
- Validate improvements with workload-specific acceptance tests.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_multiple_unoptimized.py`, `optimized_multiple_unoptimized.py`, `ai_kernel_generator.py`, `inductor_guard.py` | Composite workloads that stack several bottlenecks plus helpers for generating candidate kernels safely. |
| `baseline_pipeline_sequential.py`, `optimized_pipeline_sequential.py`, `baseline_end_to_end_bandwidth.py`, `optimized_end_to_end_bandwidth.py` | Pipeline and bandwidth case studies showing how optimizations interact across stages. |
| `baseline_integrated_kv_cache.py`, `optimized_integrated_kv_cache.py` | Integrated KV-cache demos that merge allocator, overlap, and NVLink pooling tricks. |
| `baseline_memory_standard.py`, `optimized_memory_standard.py` | Memory-focused harness verifying allocator changes at system level. |
| `baseline_training_single.py`, `optimized_training_single.py`, `test.cu`, `Makefile` | Single-device training case study plus CUDA kernels used in the final report. |
| `compare.py`, `arch_config.py`, `expectations_{hardware_key}.json` | Harness driver, architecture settings, and expectation baselines. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch20.compare
python -m cli.aisp bench list-targets --chapter ch20
python -m cli.aisp bench run --targets ch20 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m ch20.compare` emits per-stage summaries that show each optimized variant meeting or exceeding stored expectations.
- `python -m ch20.ai_kernel_generator --emit test.cu` produces CUDA kernels that compile via `nvcc` and integrate into the harness without manual edits.
- `python -m ch20.optimized_pipeline_sequential --trace` shows smooth NVTX ranges covering the entire pipeline, demonstrating overlap success.

## Notes
- `inductor_guard.py` provides convenience toggles for gating experimental kernels behind feature flags.
- `ai_kernel_generator.py` logs generated code to `artifacts/` for reproducibility; capture the log with your proof-of-benefit bundle.
