# Lab - NVFP4 Dual GEMM

## Summary
Challenge workspace for the GPUMODE NVFP4 dual-GEMM problem. It mixes baseline/optimized wrappers, official-parity local evaluation, and promotion-report AB checks.

## Problem
Challenge workspaces are easy to turn into folklore. This lab is useful only if the local evaluator, the current promoted route, and the leaderboard target stay visible at the same time.

## Baseline Path
- official/reference and baseline submission path
- correctness and challenge-semantics anchor
- much slower than the tuned route on current local measurements

## Optimized Path
- current promoted candidate route in `optimized_submission.py`
- validated primarily through official-parity local eval plus strict A/B promotion reports
- challenge workspace semantics first, generic harness story second

## Measured Delta
Current local state is best understood through two measurements:

| Measurement surface | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| Fresh official-parity local eval (`2026-03-09`, `warmup=2`, `repeats=8`, `inputs_per_repeat=20`) | `190.124 us` (`baseline_submission.py`) | pending fresh rerun | pending |
| Strict promotion A/B from `promotion_report_strict_ab.json` | `20.950 us` (prior promoted route) | `20.937 us` (`optimized_submission.py`) | `~1.00x` |

The honest takeaway is that this workspace is still a challenge-tuning loop, not a fully canonical benchmark pair. The current optimized route is close to the prior promoted route on strict A/B, and the README should say that instead of pretending every run is a giant win.

## Profiler Evidence
Use the official-parity evaluator first, then the stored promotion reports:

```bash
python -m labs.nvfp4_dual_gemm.local_eval --submission-file labs/nvfp4_dual_gemm/optimized_submission.py --reference-file labs/nvfp4_dual_gemm/reference_submission.py --warmup 2 --repeats 8 --inputs-per-repeat 20 --lock-gpu-clocks --sm-clock-mhz 1500 --json
python -m json.tool labs/nvfp4_dual_gemm/promotion_report_strict_ab.json
```

The promotion report is the artifact to trust when the per-run leaderboard numbers are too noisy to promote on their own.

## Repro Commands
```bash
python -m labs.nvfp4_dual_gemm.local_eval --submission-file labs/nvfp4_dual_gemm/baseline_submission.py --reference-file labs/nvfp4_dual_gemm/reference_submission.py --warmup 2 --repeats 8 --inputs-per-repeat 20 --lock-gpu-clocks --sm-clock-mhz 1500 --json
python -m labs.nvfp4_dual_gemm.local_eval --submission-file labs/nvfp4_dual_gemm/optimized_submission.py --reference-file labs/nvfp4_dual_gemm/reference_submission.py --warmup 2 --repeats 8 --inputs-per-repeat 20 --lock-gpu-clocks --sm-clock-mhz 1500 --json
```

## Learning Goals
- Keep the dual-GEMM challenge workspace measurable under official-parity local semantics.
- Separate "current promoted candidate" evidence from generic baseline/optimized storytelling.
- Make the promotion-report A/B flow visible in the public docs.

## Directory Layout
| Path | Description |
| --- | --- |
| `reference_submission.py`, `baseline_submission.py`, `optimized_submission.py` | Reference, baseline, and promoted candidate submission files. |
| `baseline_nvfp4_dual_gemm.py`, `optimized_nvfp4_dual_gemm.py` | Wrapper files for the benchmark-facing side of the workspace. |
| `local_eval.py`, `official_semantics_eval.py`, `promotion_report_strict_ab.json` | Official-parity evaluator plus stored A/B promotion evidence. |
| `route_sweep_verify_green.json`, `grid_sweep_verify_green.json`, `top_submission_local_screen.json` | Supporting tuning artifacts from the challenge loop. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m labs.nvfp4_dual_gemm.local_eval --submission-file labs/nvfp4_dual_gemm/baseline_submission.py --reference-file labs/nvfp4_dual_gemm/reference_submission.py --warmup 2 --repeats 8 --inputs-per-repeat 20 --lock-gpu-clocks --sm-clock-mhz 1500 --json
python -m labs.nvfp4_dual_gemm.local_eval --submission-file labs/nvfp4_dual_gemm/optimized_submission.py --reference-file labs/nvfp4_dual_gemm/reference_submission.py --warmup 2 --repeats 8 --inputs-per-repeat 20 --lock-gpu-clocks --sm-clock-mhz 1500 --json
```
- Use module invocation; direct script invocation is now re-exec'd to the same module entrypoint for compatibility.
- Treat `promotion_report_strict_ab.json` as the promotion gate when candidate deltas are small.

## Validation Checklist
- Local evaluator runs should stay verification-clean against the reference implementation.
- Promotion decisions should still be based on repeated A/B evidence, not on a single low score.

## Notes
- This is a challenge workspace first. It is benchmark-adjacent, but it is not yet a canonical harness-history lab in the same way as `labs/nvfp4_gemm` or `labs/nvfp4_group_gemm`.
