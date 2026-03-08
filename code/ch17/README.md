# Chapter 17 - Dynamic Routing & Hybrid Serving

## Summary
Blends router design, disaggregated inference, and profiling discipline so Blackwell clusters can route queries between prefill/decode pools, MoE experts, and pipeline stages without sacrificing utilization.

## Learning Goals
- Implement dynamic routers that react to TTFT, TPOT, and KV-locality metrics.
- Profile complete inference stacks (prefill + decode) under realistic synthetic loads.
- Blend pipeline parallelism with routing logic for long-context workloads.
- Document profiling steps (roofline, Nsight) specific to the routing lab.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_dynamic_routing.py`, `optimized_dynamic_routing.py`, `dynamic_routing.py`, `early_rejection.py` | Routing controllers that evolve from static heuristics to telemetry-driven admission and rejection policies. |
| `baseline_inference_full.py`, `optimized_inference_full.py`, `baseline_prefill_decode_disagg_overlap_multigpu.py`, `optimized_prefill_decode_disagg_overlap_multigpu.py`, `baseline_prefill_decode_disagg_batched_multigpu.py`, `optimized_prefill_decode_disagg_batched_multigpu.py`, `baseline_prefill_decode_disagg_ttft_multigpu.py`, `optimized_prefill_decode_disagg_ttft_multigpu.py`, `baseline_prefill_decode_disagg_tpot_long_multigpu.py`, `optimized_prefill_decode_disagg_tpot_long_multigpu.py` | End-to-end inference flows modeling separate prefill and decode pools, including overlap-focused, batched-handoff, TTFT-focused, and long-output TPOT-focused multi-GPU pairs. |
| `baseline_pipeline_parallelism.py`, `optimized_pipeline_parallelism.py` | Pipeline parallel workloads combining compute and KV-transfer scheduling. |
| `baseline_moe_router_uniform.py`, `optimized_moe_router_uniform_topology.py` | Comparable MoE router benchmark pair contrasting uniform vs topology-aware routing while keeping outputs invariant via shared expert weights. |
| `moe_router_uniform_demo.py`, `moe_router_topology_demo.py` | MoE routing demos (non-benchmark) contrasting uniform vs topology-aware expert selection. |
| `baseline_routing_static.py`, `optimized_routing_static.py` | Router variants for static/dynamic sharding decisions (comparable benchmarks). |
| `baseline_memory.py`, `optimized_memory.py`, `blackwell_profiling_guide.py` | Memory-bound case studies plus profiling guides tailored to routing workloads (use `aisp tools roofline` for roofline analysis). |
| `compare.py`, `Makefile`, `expectations_{hardware_key}.json`, `dynamo_config.yaml` | Harness entry, build rules, expectation baselines, and Dynamo config knobs. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch17.compare
python -m cli.aisp bench list-targets --chapter ch17
python -m cli.aisp bench run --targets ch17 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python optimized_dynamic_routing.py --trace` logs TTFT/TPOT trends that settle faster than the baseline's oscillations.
- `python optimized_pipeline_parallelism.py --profile minimal` shows overlapping prefill/decode segments with fewer idle bubbles.
- `python -m cli.aisp tools roofline` reproduces the documented roofline points using your latest captures.

## Notes
- `blackwell_profiling_guide.py` walks through Nsight Systems/Compute captures and interpreting roofline vs occupancy bottlenecks for routing-heavy workloads.
- `baseline_prefill_decode_disagg_overlap_multigpu.py` and `baseline_prefill_decode_disagg_batched_multigpu.py` run via torchrun and default to a 50/50 split when world size is even; override with `--prefill-ranks` (e.g., 2P1D). Use `torchrun --nproc_per_node` to choose the GPU count.
- The disaggregated prefill/decode baselines use per-request blocking handoff with per-request sync/barrier to model naive scheduling; optimized counterparts batch per group or send contiguous KV/seed slabs to overlap or boost throughput.
