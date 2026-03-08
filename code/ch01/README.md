# Chapter 1 - Performance Fundamentals

## Summary
Establishes the baseline benchmarking discipline with a simple training-loop goodput benchmark and a small CUDA GEMM case study. The goal is to ground later optimizations in repeatable measurement, equivalent workloads, and verifiable outputs.

## Learning Goals
- Profile a minimal PyTorch training loop with the shared harness and reason about throughput vs latency.
- Apply basic optimizations (FP16 + fused microbatches) without changing the algorithmic workload.
- Compare hand-written GEMM kernels in batched vs. strided forms to understand arithmetic intensity.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_performance.py`, `optimized_performance.py` | Goodput-focused training loop pair comparing FP32 eager vs FP16 + fused microbatches (batch fusion). |
| `baseline_gemm.cu`, `optimized_gemm_batched.cu`, `optimized_gemm_strided.cu` | CUDA GEMM variants (single, batched, strided) used to illustrate launch amortization and memory coalescing. |
| `compare.py`, `workload_config.py`, `arch_config.py`, `expectations_{hardware_key}.json` | Harness entrypoint, workload shapes, architecture overrides, and stored expectation thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch01.compare
python -m cli.aisp bench list-targets --chapter ch01
python -m cli.aisp bench run --targets ch01 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m ch01.compare` reports the chapter baseline/optimized training loop pair through the shared harness with consistent workloads.
- Running `make && ./baseline_gemm_sm100` vs `./optimized_gemm_batched_sm100` shows a substantial drop in launch count and total runtime.

## Notes
- `requirements.txt` pins lightweight extras (Typer, tabulate) used by helper scripts.
- `Makefile` builds the CUDA GEMM binaries with SM-specific suffixes for quick diffing.
