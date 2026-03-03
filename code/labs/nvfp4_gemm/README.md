# Lab NVFP4 GEMM (Leaderboard 597)

This lab is the dedicated workspace for tuning the NVFP4 GEMM challenge (`leaderboard 597`) on B200.
It contains both:
- CUDA-binary A/B benchmarking flow (`baseline_nvfp4_gemm.cu` vs `optimized_nvfp4_gemm.cu`)
- Submission-style flow aligned with the GPUMODE reference task (`task.py`, `reference_submission.py`, `optimized_submission.py`)

## Leaderboard + Reference

- Leaderboard rankings: <https://www.gpumode.com/leaderboard/597?tab=rankings>
- Reference implementation tab: <https://www.gpumode.com/leaderboard/598?tab=reference>

## Files

- `baseline_nvfp4_gemm.cu`: Baseline CUDA implementation.
- `optimized_nvfp4_gemm.cu`: Optimized CUDA implementation (currently includes shape-specific kernel dispatch).
- `baseline_nvfp4_gemm.py` / `optimized_nvfp4_gemm.py`: Harness wrappers for `bench` targets.
- `local_eval.py`: Interleaved A/B rerank driver for CUDA-binary path (with verify and clock-lock options).
- `task.py`, `utils.py`, `reference_submission.py`: Reference task contract + reference implementation mirror.
- `optimized_submission.py`: Current submission candidate.
- `local_eval_submission.py`: Leaderboard-style submission evaluator (geomean score in seconds/us).
- `local_eval_official597.py`: Official-semantics evaluator matching `eval_better_bench.py` leaderboard behavior.
- `sweep_case0_official.py`: Case0-only structural sweep runner on top of `local_eval_official597.py`.

## Quick Start

Build and run interleaved A/B rerank locally:

```bash
python labs/nvfp4_gemm/local_eval.py --pairs 8 --verify --json-out /tmp/nvfp4_gemm_ab.json
```

Run leaderboard-style submission eval:

```bash
python labs/nvfp4_gemm/local_eval_submission.py \
  --submission-file labs/nvfp4_gemm/optimized_submission.py \
  --verify --repeats 12 --inputs-per-repeat 50
```

Run official-semantics submission eval (recommended for tuning):

```bash
python labs/nvfp4_gemm/local_eval_official597.py \
  --submission-file labs/nvfp4_gemm/optimized_submission.py \
  --json-out /tmp/nvfp4_official597.json
```

Run a case0-only official-semantics sweep for case0 structural variants:

```bash
python labs/nvfp4_gemm/sweep_case0_official.py \
  --submission-file labs/nvfp4_gemm/optimized_submission.py \
  --variants v4_default,v4_n64_s1,v4_n64_split2,v4_n128_s1,v4_m64_n128_split2_s8,v4_m64_n128_split1_s8,v4_m64_n64_split2_s8,v4_m64_n64_split1_s8,v4_m64_n128_bk512_s3,v3b
```

Run via benchmark harness target discovery:

```bash
python -m cli.aisp bench list-targets --chapter labs/nvfp4_gemm
python -m cli.aisp bench run --targets labs/nvfp4_gemm:optimized_nvfp4_gemm --profile minimal
```

## Notes

- This lab is separate from `ch09` so challenge-specific tuning and artifacts stay isolated.
- The CUDA binaries emit `TIME_MS:` as geomean over challenge shapes; `local_eval.py` uses that value for A/B summaries.
- `local_eval_submission.py` tracks against `TOP_SCORE_SECONDS_597 = 9.981888843481874e-06` (queried from GPUMODE API on February 28, 2026).
- `optimized_submission.py` supports `AISP_NVFP4_CASE0_VARIANT` for case0 routing:
  `v4_default`, `v4_n64_s1`, `v4_n64_split2`, `v4_n128_s1`,
  `v4_m64_n128_split2_s8`, `v4_m64_n128_split1_s8`, `v4_m64_n64_split2_s8`,
  `v4_m64_n64_split1_s8`, `v4_m64_n128_bk512_s3`, `v3b`.
- Current default is `AISP_NVFP4_CASE0_VARIANT=v3b` for correctness stability in long official-semantics loops.
