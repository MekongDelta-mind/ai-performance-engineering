# NVFP4 Dual GEMM (Leaderboard 598)

This lab is for iterating on the GPU MODE NVFP4 dual-GEMM challenge (`leaderboard 598`) locally on B200.

## Files

- `reference_submission.py`: Official reference implementation from `gpu-mode/reference-kernels`.
- `baseline_submission.py`: Baseline wrapper using the reference kernel.
- `optimized_submission.py`: Fastest locally-screened competitive submission variant (sourced from leaderboard submission `379817` and adapted to run quietly in local eval).
- `local_eval.py`: Local leaderboard-style evaluator with correctness checks and clock locking.

## Run

```bash
python labs/nvfp4_dual_gemm/local_eval.py \
  --submission-file labs/nvfp4_dual_gemm/optimized_submission.py \
  --reference-file labs/nvfp4_dual_gemm/reference_submission.py \
  --warmup 3 --repeats 20
```

JSON output:

```bash
python labs/nvfp4_dual_gemm/local_eval.py --json
```

## Notes

- Current leaderboard target (from `GPUMODE/kernelbot-data`, queried on February 28, 2026):
  - `best_score_seconds = 1.2913403524642259e-05` (lower is better)
- Current local best with this lab (B200, lock clocks on, warmup=2, repeats=16, inputs-per-repeat=50):
  - `score_seconds = 2.0153966462467644e-05` (`20.153966 us`)
- Local score is highly sensitive to CUTLASS/CUDA toolchain and runtime behavior.
