# Chapter 5 - Storage and IO Optimization

## Summary
Focuses on feeding GPUs efficiently: tune DataLoader workers, vectorize preprocessing, overlap IO with compute, and adopt GPUDirect Storage when NVMe traffic becomes the bottleneck.

## Learning Goals
- Detect IO stalls via harness metrics and restructure pipelines to keep GPUs busy.
- Tune PyTorch DataLoader knobs (workers, prefetch, pinned memory) for large-batch training.
- Evaluate GPUDirect Storage paths vs traditional CPU-mediated reads.
- Benchmark remote storage and distributed data reading strategies.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_storage_cpu.py`, `optimized_storage_cpu.py` | Single-node dataloader comparison covering worker count, pinned memory, and caching strategies. |
| `baseline_vectorization.py`, `optimized_vectorization.py` | Vectorized parsing and memory-map examples that remove Python loops from preprocessing. |
| `baseline_ai.py`, `optimized_ai.py`, `storage_io_optimization.py` | LLM-style token pipelines showcasing overlapping compute with streaming reads and prefetch. |
| `baseline_distributed.py`, `optimized_distributed.py` | Single-GPU sum vs optional distributed all-reduce fallback. |
| `baseline_distributed_multigpu.py`, `optimized_distributed_multigpu.py` | Multi-GPU reduction baseline (CPU staging) vs GPU-side reduce_add. |
| `gds_cufile_minimal.py`, `gpudirect_storage_example.py` | GPUDirect Storage samples for verifying cuFile setup, buffer alignment, and throughput. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json` | Harness entrypoint plus expectation baselines for spotting regressions. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch05.compare
python -m cli.aisp bench list-targets --chapter ch05
python -m cli.aisp bench run --targets ch05 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python baseline_storage_cpu.py --inspect` exposes CPU wait time > GPU time; `optimized_storage_cpu.py` reverses the ratio with >=80% GPU utilization.
- `python -m ch05.gds_cufile_minimal /tmp/gds_test_file.bin 1073741824 --generate` sustains multi-GB/s throughput when `/etc/cufile.json` is configured and NVMe advertises GPUDirect support.
- `python -m ch05.compare` shows optimized_ai eliminating CPU-side preprocessing from the critical path.

## Notes
- GPUDirect scripts fall back to host-mediated reads when `libcufile.so` is unavailable, making it safe to run on dev laptops.
- `requirements.txt` captures the limited extra deps (like `lmdb`) needed for the dataset shims.
