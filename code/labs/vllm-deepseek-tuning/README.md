# vLLM DeepSeek Tuning Harness (GB300 blog-aligned, production-ready)

This lab reproduces and extends the benchmark structure from:
https://blog.vllm.ai/2026/02/13/gb300-deepseek.html

Location:
`/Users/admin/dev/ai-perf/ai-performance-engineering/code/labs/vllm-deepseek-tuning`

## Goals

- Reproduce blog-style scenarios on available GPUs.
- Keep experiment shapes consistent (ISL/OSL/concurrency sweeps).
- Produce auditable artifacts (raw logs + JSON + plots + markdown report).
- Make reruns easy and deterministic.

## What is implemented

- Scenarios:
  - `prefill_only` (ISL=2k, OSL=1)
  - `mixed_short_64` (ISL=2k, OSL=64)
  - `mixed_short_128` (ISL=2k, OSL=128)
  - `mixed_moderate_1k` (ISL=2k, OSL=1000)
- Run variants:
  - DeepSeek-V3.2 NVFP4: TP2 / TP4
  - DeepSeek-R1 NVFP4: TP2 / EP2
  - DeepSeek-R1 TP2 + MTP(1)
- Outputs:
  - raw benchmark logs (`results/*.raw.txt`)
  - structured run records (`results/*.json`)
  - comparison plots (`plots/*.png`)
  - report pack (`reports/summary.csv`, `reports/report.md`)

## Project layout

- `configs/benchmark_matrix.yaml` — experiment matrix + paths + profile settings
- `configs/smoke_tiny.yaml` — tiny open-model config for end-to-end harness sanity checks
- `scripts/run_matrix.py` — serve/bench orchestration
- `scripts/plot_results.py` — chart generation
- `scripts/report_results.py` — summary CSV + blog-style markdown report
- `scripts/vllm_docker.sh` — Docker-backed `vllm` command wrapper (keeps host torch untouched)
- `scripts/teardown.sh` — stop serving + optional artifact purge
- `Makefile` — standard entrypoints

## Prerequisites

- Python 3.10+
- `vllm` CLI available on PATH (or use Docker `vllm/vllm-openai:cu130-nightly` via `--vllm-cmd`)
- CUDA/NVIDIA runtime healthy on benchmark node
- Model + tokenizer checkpoints available locally

Recommended to mirror blog as closely as possible:
- vLLM v0.14.1
- CUDA 13.0

## Configure paths first

Edit `configs/benchmark_matrix.yaml`:

- `paths.tokenizer` must be a valid local path (or HF tokenizer ID) on the GPU node
- confirm model IDs are pullable (or replace with local paths)
- adjust `results_dir` only if you want artifacts elsewhere
- set `global.vllm_cmd` only if you need a non-default vLLM command

Default `results_dir` is `../results` (which lands in the lab's `results/` directory) and is resolved relative to the config file.
You can override without editing config:

```bash
python3 scripts/run_matrix.py \
  --config configs/benchmark_matrix.yaml \
  --tokenizer /path/to/DeepSeek-V3.2 \
  --results-dir /tmp/vllm-deepseek-results \
  --base-url http://127.0.0.1:8000 \
  --vllm-cmd "vllm"
```

If host `vllm` is incompatible with host `torch`, use Docker without changing `torch`:

```bash
python3 scripts/run_matrix.py \
  --config configs/benchmark_matrix.yaml \
  --vllm-cmd ./scripts/vllm_docker.sh
```

## Runbook

From this lab directory:

```bash
cd /Users/admin/dev/ai-perf/ai-performance-engineering/code/labs/vllm-deepseek-tuning
```

### 1) Smoke test

```bash
make smoke
```

### 1b) End-to-end infra sanity (tiny open model)

```bash
make smoke-tiny      # host vllm
make smoke-docker    # dockerized vllm (recommended when host vllm is mismatched)
```

### 2) Full benchmark matrix

```bash
make full
```

### 3) Generate plots + reports

```bash
make artifacts
```

### 4) Teardown

```bash
make teardown               # stops vllm serve
make purge                  # stops + removes results/plots/reports
```

## Report quality (blog-aligned + improved)

`reports/report.md` includes:

- V3.2 TP2 vs TP4 comparison
- R1 TP2 vs EP2 comparison
- MTP on/off comparison
- R1 vs V3.2 prefill ratio
- per-run per-scenario bests

This goes beyond raw output by computing concise deltas and direct conclusions.

## Notes on interpretation

- Absolute throughput is hardware-dependent.
- Relative comparisons (TP/EP/MTP deltas, model-vs-model ratios) are the most transferable.
- Failed run records are preserved for auditability but excluded from best-of metrics.

## Design choices

- Scenario shape fixed to match the blog patterns.
- Concurrency profiles:
  - `smoke`: fast validation
  - `full`: broad sweep
- Every run writes both raw and parsed artifacts.
- Teardown script provides deterministic cleanup.
