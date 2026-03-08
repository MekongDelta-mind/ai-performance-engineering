# Lab - Real-World Model Optimizations

## Summary
Applies the course-wide optimization patterns to representative models (Llama 3.1 8B, DeepSeek-R1 MoE, GPT-4-style) so you can practice end-to-end tuning on Blackwell and Grace-Blackwell hardware.

## Problem
Microbenchmarks are useful, but they can hide whether the repo's optimizations still matter on a real model path. This lab is the end-to-end check.

## Baseline Path
- representative model skeletons with conservative serving/training defaults
- enough realism to surface KV-cache, routing, and compile effects
- intentionally simpler than production deployment code so the optimization deltas stay readable

## Optimized Path
- torch.compile and fused attention where they help
- topology-aware and memory-aware configuration choices
- the same benchmark harness contract as the lower-level labs

## Measured Delta
Current validated result from `artifacts/runs/20260302_full_strict_all_singlegpu/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `llama_3_1_8b` | `13.143 ms` | `5.274 ms` | `2.49x` |

This is the right lab to use when you want to sanity-check that the lower-level wins still add up on a model-shaped workload.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/real_world_models:llama_3_1_8b --profile deep_dive --single-gpu
```

That path keeps the same evidence model as the rest of the repo: baseline/optimized timing, validation, and profiler artifacts in one run tree.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/real_world_models
python -m cli.aisp bench run --targets labs/real_world_models --profile minimal
python labs/real_world_models/llama_3_1_8b_optimization.py --seq-length 8192 --use-compile
```

## Learning Goals
- Exercise attention, MoE, and memory optimizations on realistic architectures instead of toy kernels.
- Use the benchmark harness to collect reproducible throughput/latency metrics across models.
- Track expert balance, routing entropy, and KV-cache pressure while iterating on serving choices.
- Compare FP8/FP16, torch.compile, and topology-aware placements without changing source code.

## Directory Layout
| Path | Description |
| --- | --- |
| `llama_3_1_8b_optimization.py` | Single-node 8B walkthrough with `torch.compile`, FlexAttention, and Flash SDPA toggles. |
| `deepseek_r1_moe_optimization.py` | 64-expert top-6 routing demo with balance/Gini/entropy metrics and auxiliary loss. |
| `gpt4_architecture_optimization.py` | GPT-4-style MoE + context-parallel sketch with FP8 support and memory estimation. |
| `__init__.py` | Exports harness targets for the CLI. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering
python -m cli.aisp bench list-targets --chapter labs/real_world_models
python -m cli.aisp bench run --targets labs/real_world_models --profile minimal
# Direct runs
python labs/real_world_models/llama_3_1_8b_optimization.py --seq-length 8192 --use-compile
python labs/real_world_models/deepseek_r1_moe_optimization.py --num-experts 64 --top-k 6 --batch-size 4
python labs/real_world_models/gpt4_architecture_optimization.py --seq-length 8192 --context-parallel
```
- Override per-model flags via `--target-extra-arg labs/real_world_models:<target>="--flag value"` when using the harness.

## Validation Checklist
- `llama_3_1_8b_optimization.py` sustains ~20K tokens/sec on B200 with `--use-compile` enabled and stays memory-efficient at 8K+ context.
- `deepseek_r1_moe_optimization.py` reports balanced experts (Gini < 0.2) and stable router entropy across batches.
- `gpt4_architecture_optimization.py` runs the context-parallel path without OOM on appropriately sized clusters; memory estimates match the printed budget.
- Harness runs emit comparable baseline/optimized timings for every target without manual wiring.

## Notes
- These scripts are intentionally weight-light sketches for benchmarking; swap in real checkpoints to validate production settings.
- Hardware expectations: B200/GB200 for best results; GPT-4-scale examples assume 24+ GPUs with NVLink/NVL fabrics.
- Metrics (balance loss, entropy, KV cache) are emitted alongside throughput so you can gate deployments with more than raw speed.
