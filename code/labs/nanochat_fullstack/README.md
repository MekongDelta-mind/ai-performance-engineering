# Lab - NanoChat Fullstack

## Summary
Wraps the NanoChat full-stack tree with a clean harness benchmark pair so the repo can talk about a real end-to-end inference stack with measured baseline vs optimized deltas, not just kernels.

## Problem
Full-stack LLM projects are easy to describe in product terms and hard to benchmark cleanly. This lab keeps a narrow baseline/optimized inference pair inside the larger NanoChat tree so the performance story stays measurable.

## Baseline Path
- slower NanoChat inference path
- end-to-end reference inside the same full-stack project
- useful for checking whether the optimized path is buying real latency reduction

## Optimized Path
- optimized NanoChat inference path
- same harness contract and verification expectations
- intended to represent the practical serving-side improvements, not just a kernel microbench

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `nanochat_inference` | `122.975 ms` | `67.621 ms` | `1.82x` |

That is the useful local story: NanoChat is still a full-stack project, but the repo now has a concrete measured inference delta for it instead of leaving the performance claim buried in a much larger README.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/nanochat_fullstack:nanochat_inference --profile deep_dive --single-gpu
```

Use the deep-dive path when you want Nsight evidence for the inference stack. Keep the `speedrun.sh` story separate from the benchmark pair; they answer different questions.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/nanochat_fullstack
python -m cli.aisp bench run --targets labs/nanochat_fullstack:nanochat_inference --profile minimal
python -m cli.aisp bench verify -t labs/nanochat_fullstack:nanochat_inference
```

## Learning Goals
- Keep a real full-stack LLM project in the benchmark story, not just microkernels.
- Benchmark NanoChat inference as a clean baseline/optimized pair inside the larger tree.
- Point readers at the broader project context without losing the measured harness story.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_nanochat_inference.py`, `optimized_nanochat_inference.py` | Harness benchmark pair for NanoChat inference. |
| `benchmark_incremental_optimizations.py` | Incremental benchmarking helper inside the NanoChat tree. |
| `speedrun.sh`, `run1000.sh`, `README_FAST.md` | Broader NanoChat quick-start and end-to-end project entrypoints. |
| `nanochat/`, `scripts/`, `tasks/`, `tests/` | Core NanoChat project tree and operational helpers. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/nanochat_fullstack
python -m cli.aisp bench run --targets labs/nanochat_fullstack --profile minimal
```
- Targets follow the `labs/nanochat_fullstack:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/nanochat_fullstack:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/nanochat_fullstack:nanochat_inference --profile minimal` should keep the optimized path ahead under the harness contract.
- `python -m cli.aisp bench verify -t labs/nanochat_fullstack:nanochat_inference` should stay green before any performance claim is accepted.

## Project Context
NanoChat is intentionally bigger than a single benchmark pair. The point of this lab entry is to give the repo one clean performance anchor inside that tree, not to replace the broader NanoChat project documentation.

- Use [README_FAST.md](/home/cfregly/ai-performance-engineering/code/labs/nanochat_fullstack/README_FAST.md) for the faster end-to-end project walkthrough.
- Use [speedrun.sh](/home/cfregly/ai-performance-engineering/code/labs/nanochat_fullstack/speedrun.sh) when you want the broader "train and talk to a small model" experience.
- Use [rustbpe/README.md](/home/cfregly/ai-performance-engineering/code/labs/nanochat_fullstack/rustbpe/README.md) for the tokenizer-specific component work.

## Notes
- This README focuses on the repo's benchmarked NanoChat story. Use `README_FAST.md` and the project scripts when you want the broader training/serving walkthrough.
- The decode microbenchmarks live separately in `labs/decode_optimization`; this lab is the broader inference-stack companion.
