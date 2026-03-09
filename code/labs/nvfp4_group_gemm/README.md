# Lab - NVFP4 Grouped GEMM

## Summary
Explores grouped-GEMM routing and schedule variants across multiple cases so you can see where the grouped NVFP4 path is actually winning and where it is merely legal.

## Problem
Grouped GEMM tuning is noisy and easy to overclaim. This lab keeps the case routing explicit and benchmarked so promotions are based on repeated verified wins instead of one-off lows.

## Baseline Path
- per-case baseline grouped GEMM paths
- stable routing reference for cases 0-3
- useful for showing which grouped shapes are hard versus easy

## Optimized Path
- per-case tuned NVFP4 grouped GEMM variants
- same grouped workloads, but explicit schedule/routing choices
- designed to keep promotions tied to repeated verify and ABAB checks

## Measured Delta
Representative strict all-case results from `artifacts/runs/20260302_rerun_all_labschapters_strict/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `nvfp4_group_gemm_case0` | `8.361 ms` | `4.180 ms` | `2.00x` |
| `nvfp4_group_gemm_case1` | `10.285 ms` | `1.422 ms` | `7.23x` |
| `nvfp4_group_gemm_case2` | `3.708 ms` | `1.087 ms` | `3.41x` |
| `nvfp4_group_gemm_case3` | `3.348 ms` | `1.117 ms` | `3.00x` |

Case 1 is the biggest local winner, but the lab is most valuable because it keeps all four cases visible instead of letting one good case stand in for the whole grouped-GEMM story.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile deep_dive --single-gpu
```

Use the harness artifacts for schedule attribution, then use the router/ABAB tooling for promotion decisions. The benchmark pair tells you the shape of the win; the tuning scripts decide whether a default should actually move.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/nvfp4_group_gemm
python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile minimal
```

## Learning Goals
- Keep grouped-GEMM tuning grounded in repeated verified case-by-case evidence.
- Benchmark the promoted routes for all four grouped cases under one harness family.
- Separate exploration scripts from the regression-tracked benchmark defaults.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_nvfp4_group_gemm_case0.py` ... `baseline_nvfp4_group_gemm_case3.py` | Per-case baseline grouped-GEMM entrypoints. |
| `optimized_nvfp4_group_gemm_case0*.py` ... `optimized_nvfp4_group_gemm_case3*.py` | Per-case tuned grouped-GEMM variants. |
| `WORKLOG.md`, `custom_cuda_submission.py`, `cutlass_extension.py` | Tuning log and implementation plumbing for the promoted routes. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/nvfp4_group_gemm
python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile minimal
```
- Targets follow the `labs/nvfp4_group_gemm:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/nvfp4_group_gemm:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile minimal` should keep all promoted case routes verification-clean.
- Default changes should still be gated by the stricter ABAB/verify process documented in the codebase notes, not by a single benchmark run.

## Notes
- This lab is intentionally stricter than a normal benchmark pair because grouped-GEMM route tuning is unusually noise-prone.
