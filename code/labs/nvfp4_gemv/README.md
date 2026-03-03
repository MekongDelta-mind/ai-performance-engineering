# Lab NVFP4 GEMV

This lab is the dedicated workspace for the GPUMODE `nvfp4_gemv` challenge (leaderboard 595).

## Leaderboard + Reference

- Leaderboard: <https://www.gpumode.com/leaderboard/595?tab=rankings>
- Reference: <https://www.gpumode.com/leaderboard/595?tab=reference>

## Files

- `reference_submission.py`: Reference implementation mirror.
- `baseline_submission.py`: Baseline local submission.
- `optimized_submission.py`: Current optimized local submission.
- `local_eval.py`: Official-parity harness wrapper that stages and runs upstream `problems/nvidia/eval.py` in `leaderboard` mode.
- `task.py` / `utils.py`: Typing and helper utilities.

## Quick Start (Official Semantics)

```bash
python labs/nvfp4_gemv/local_eval.py \
  --submission-file labs/nvfp4_gemv/baseline_submission.py \
  --json

python labs/nvfp4_gemv/local_eval.py \
  --submission-file labs/nvfp4_gemv/optimized_submission.py \
  --json
```

To run Popcorn benchmark mode (when service is available):

```bash
python -m popcorn.main submit \
  --leaderboard nvfp4_gemv \
  --path labs/nvfp4_gemv/optimized_submission.py \
  --mode benchmark
```

## Latest Local Results (B200)

- Harness: exact upstream `eval.py` leaderboard semantics (`recheck=True`, adaptive repeat stopping, `clear_l2_cache` each repeat).
- GPU lock mode: app clocks via harness (`sm=1500 MHz` by default).
- Baseline (`baseline_submission.py`): `~203.71 us` geomean.
- Optimized (`optimized_submission.py`): `~68.21 us` geomean (best-of-4 repeated official-parity runs).

## Notes

- Current GPUMODE service responses for submit are intermittently returning `503 Offline for Maintenance`.
- Current default route uses `gemm_v3b` with `AISP_NVFP4_GEMV_CASE1_N_EFF=96` for the `k=7168` case.
