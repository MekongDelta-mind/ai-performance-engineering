# Lab - FlashAttention Gluon

## Summary
Benchmarks a FlashAttention-style optimized path against a simpler attention reference so the local Gluon-flavored integration stays measured and honest.

## Problem
Attention-stack integrations can look "fast" because the benchmark is fuzzy. This lab keeps the pair narrow so you can see whether the Gluon-oriented optimized path really buys anything on this stack.

## Baseline Path
- simple attention reference path
- correctness anchor for the optimized implementation
- no fused fast-path assumptions

## Optimized Path
- FlashAttention-style optimized path
- same workload and harness contract
- focused on local integration cost/benefit, not a synthetic peak score

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `flashattention_gluon` | `0.205 ms` | `0.154 ms` | `1.33x` |

This is a modest but real backend/path win. The useful part is that the result stays measured and reproducible instead of being hidden in a broader model benchmark.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/flashattention_gluon:flashattention_gluon --profile deep_dive --single-gpu
```

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/flashattention_gluon
python -m cli.aisp bench run --targets labs/flashattention_gluon:flashattention_gluon --profile minimal
```

## Learning Goals
- Keep the local FlashAttention/Gluon integration benchmarked as a clean pair.
- Measure backend-path value without mixing in unrelated model-level effects.
- Use a small, stable attention benchmark as an integration health signal.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flashattention_gluon.py`, `optimized_flashattention_gluon.py` | Baseline and optimized harness entrypoints. |
| `flashattention_gluon_common.py` | Shared workload setup and helper code. |
| `expectations_{hardware_key}.json` | Regression thresholds for the lab. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/flashattention_gluon
python -m cli.aisp bench run --targets labs/flashattention_gluon --profile minimal
```
- Targets follow the `labs/flashattention_gluon:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/flashattention_gluon:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/flashattention_gluon:flashattention_gluon --profile minimal` should keep the optimized path ahead on validated hardware.

## Notes
- Treat this as an integration-health benchmark more than as a giant architectural headline win.
