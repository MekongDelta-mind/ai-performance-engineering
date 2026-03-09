# Lab - KV Cache Compression

## Summary
Tests whether compressing the KV cache is worth it for this workload, instead of assuming lower memory footprint automatically means better serving latency.

## Problem
KV-cache compression is attractive because the memory story is obvious, but the latency story often is not. This lab exists to keep those two questions separate.

## Baseline Path
- uncompressed KV cache path
- simple latency/memory reference
- no compression overhead in the hot path

## Optimized Path
- compressed KV cache representation
- same benchmark harness and validation contract
- tests whether the memory tradeoff is actually latency-positive here

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `kv_cache` | `6066.040 ms` | `5897.083 ms` | `1.03x` |

The important takeaway is restraint: the compressed path helps, but only slightly on this workload. This is exactly the kind of lab where a clean benchmark pair prevents an overclaim.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/kv_cache_compression:kv_cache --profile deep_dive --single-gpu
```

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/kv_cache_compression
python -m cli.aisp bench run --targets labs/kv_cache_compression:kv_cache --profile minimal
```

## Learning Goals
- Measure the latency cost/benefit of KV-cache compression under the harness contract.
- Keep memory-saving and latency-saving claims distinct.
- Make it easy to inspect whether compression overhead dominates the win.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_kv_cache.py`, `optimized_kv_cache_nvfp4.py` | Baseline and compressed KV-cache benchmark pair. |
| `kv_cache_common.py` | Shared workload setup. |
| `tma_prefetch_extension.py`, `tma_prefetch_ext.cu` | Extension code supporting the optimized path. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/kv_cache_compression
python -m cli.aisp bench run --targets labs/kv_cache_compression --profile minimal
```
- Targets follow the `labs/kv_cache_compression:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/kv_cache_compression:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/kv_cache_compression:kv_cache --profile minimal` should keep the compressed path verification-clean and modestly ahead on this hardware.

## Notes
- This is a good lab for demonstrating that some memory optimizations are valuable mostly for capacity, not for giant latency wins.
