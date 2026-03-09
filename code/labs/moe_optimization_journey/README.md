# Lab - MoE Optimization Journey

## Summary
Packages a staged MoE optimization story from naive execution to quantized/padded fast paths so you can measure which step is actually doing the work.

## Problem
MoE optimization is often told as a narrative, not a benchmarked sequence. This lab keeps the sequence explicit so you can see which stage of the journey is providing the real win.

## Baseline Path
- naive MoE execution path
- simple correctness reference
- useful for showing how expensive unstructured expert execution can be

## Optimized Path
- staged optimized MoE path with batching/layout/scheduling improvements
- separate padded/quantized route for a more production-like fast path
- designed to attribute wins to concrete optimization steps

## Measured Delta
Representative strict results from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `moe` | `41.938 ms` | `1.217 ms` | `34.47x` |
| `moe_pad_quant` | `4.681 ms` | `1.790 ms` | `2.62x` |

The spread is useful. The big win is in the core MoE path, while the padded/quantized lane is a smaller, still-real follow-on improvement.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/moe_optimization_journey:moe --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/moe_optimization_journey:moe_pad_quant --profile deep_dive --single-gpu
```

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/moe_optimization_journey
python -m cli.aisp bench run --targets labs/moe_optimization_journey --profile minimal
```

## Learning Goals
- Show a stepwise MoE optimization story with measured deltas instead of vague progression.
- Keep the naive path, batched path, and padded/quantized path benchmarked under one roof.
- Make it obvious which optimization stage is worth carrying forward.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_moe.py`, `baseline_moe_pad_quant.py` | Naive/reference entrypoints. |
| `level0_naive.py` through `level6_full_stack.py` | Incremental optimization stages used by the journey. |
| `moe_benchmark.py` | Shared benchmark harness layer for the staged MoE path. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/moe_optimization_journey
python -m cli.aisp bench run --targets labs/moe_optimization_journey --profile minimal
```
- Targets follow the `labs/moe_optimization_journey:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/moe_optimization_journey:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/moe_optimization_journey --profile minimal` should keep both the core MoE and pad/quant targets green.
- Deep-dive runs should make the kernel/layout win attributable to the staged path rather than only to end-to-end timing.

## Notes
- This lab is a good example of how the repo should teach optimization: staged, benchmarked, and profiler-backed.
