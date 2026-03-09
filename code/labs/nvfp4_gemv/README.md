# Lab - NVFP4 GEMV

## Summary
Challenge workspace for the GPUMODE NVFP4 GEMV problem with exact official leaderboard semantics preserved in a local evaluator.

## Problem
Challenge workspaces need an honest measurement surface. This lab keeps the official leaderboard semantics explicit so the optimized route is judged against the real baseline, not a toy proxy.

## Baseline Path
- `baseline_submission.py`
- official-parity local eval path
- correctness and challenge-semantics anchor

## Optimized Path
- `optimized_submission.py`
- same official eval semantics and clock-locking path
- challenge-oriented route with case-specific tuning

## Measured Delta
Representative official-parity local results from `labs/nvfp4_gemv/official_eval_*_20260228.json`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `nvfp4_gemv` | `203.711 us` | `68.206 us` | `2.99x` |

That is a real challenge-workspace win, and the important part is that it comes from the official evaluator semantics rather than from an easier local proxy.

## Profiler Evidence
```bash
python -m labs.nvfp4_gemv.local_eval --submission-file labs/nvfp4_gemv/baseline_submission.py --json
python -m labs.nvfp4_gemv.local_eval --submission-file labs/nvfp4_gemv/optimized_submission.py --json
```

The evaluator produces per-case means plus the official aggregate score, which is the right artifact to compare in this workspace.

## Repro Commands
```bash
python -m labs.nvfp4_gemv.local_eval --submission-file labs/nvfp4_gemv/baseline_submission.py --json
python -m labs.nvfp4_gemv.local_eval --submission-file labs/nvfp4_gemv/optimized_submission.py --json
```

## Learning Goals
- Keep the GEMV challenge workspace aligned with the official evaluator semantics.
- Make the baseline vs optimized submission delta visible with real stored artifacts.
- Prevent local-tuning wins from being claimed without official-parity evidence.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_submission.py`, `optimized_submission.py`, `reference_submission.py` | Submission files for the challenge workspace. |
| `baseline_nvfp4_gemv.py`, `optimized_nvfp4_gemv.py` | Wrapper files for benchmark-facing integration. |
| `local_eval.py`, `official_eval_baseline_20260228.json`, `official_eval_optimized_20260228.json` | Official-parity local evaluator and stored result artifacts. |
| `task.py`, `utils.py` | Challenge helpers and task definitions. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m labs.nvfp4_gemv.local_eval --submission-file labs/nvfp4_gemv/baseline_submission.py --json
python -m labs.nvfp4_gemv.local_eval --submission-file labs/nvfp4_gemv/optimized_submission.py --json
```
- Use module invocation; direct script invocation is now re-exec'd to the same module entrypoint for compatibility.
- When the Popcorn service is healthy, compare these local official-parity results against benchmark-mode submissions rather than against ad hoc local timings.

## Validation Checklist
- The official-parity local evaluator should pass correctness and emit per-case timing plus aggregate score.
- Any promoted route should stay explainable in terms of the official evaluator output, not only custom local scripts.

## Notes
- This lab already has a cleaner local evidence story than `nvfp4_dual_gemm` because the stored official baseline and optimized reports are checked in.
