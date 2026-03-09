# Lab - Python Concurrency Playbook

## Summary
A control-plane lab for Python concurrency work: bounded queues, retries, cancellation, idempotency, hybrid async/process pipelines, and the operational invariants that keep them correct.

## Problem
Concurrency bugs usually look like performance bugs until you inspect the invariants. This lab is here to teach the control-plane discipline directly: bounded pressure, explicit failure handling, and deterministic terminal state.

## What This Lab Is
- a scenario and playbook lab
- multiple runnable reference scripts
- focused on correctness plus measurable throughput/latency behavior

It is **not** currently a single baseline/optimized benchmark pair, and the README should say that plainly.

## What A Proper Benchmark Pair Would Look Like
If we decide to productize this as a harness target later, the clean shape would be something like:

- `baseline_sync_pipeline.py`: serial or poorly bounded control path
- `optimized_hybrid_pipeline.py`: bounded async + process-pool pipeline

with a fixed JSON workload, invariant checks (`one terminal status per item`, ordered output, retry accounting), and measured outputs such as throughput plus p95/p99. That would be a new benchmark pair, not just a rename of the current playbook scripts.

## Learning Goals
- Teach practical concurrency design under production-style constraints.
- Keep correctness invariants visible instead of treating them as afterthoughts.
- Provide runnable drills for async I/O, CPU work, retries, cancellation, and hybrid pipelines.

## Directory Layout
| Path | Description |
| --- | --- |
| `all_in_one_pipeline.py` | Single-file reference for bounded queues, retries, dedupe, timeout handling, and hybrid async/process execution. |
| `taskrun_round1_asyncio.py`, `taskrun_round2_controls.py`, `taskrun_round3_idempotency.py` | Staged drills that build the playbook incrementally. |
| `hybrid_three_stage_pipeline.py`, `executors_cpu_vs_io.py`, `gil_demo.py` | Focused experiments for workload classification and executor behavior. |
| `ADVANCED_SCENARIOS.md`, `SCENARIO_QA.md`, `QUICK_REFERENCE_GUIDE.md` | Interview/playbook-oriented docs that explain the patterns and failure modes. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python labs/python_concurrency/all_in_one_pipeline.py --input labs/python_concurrency/sample_all_in_one_items.json --stage-a-workers 3 --stage-b-workers 2 --stage-c-workers 2 --queue-size 4 --rps 12 --fetch-inflight 3 --write-inflight 2 --fetch-timeout-ms 200 --write-timeout-ms 200 --cpu-timeout-ms 1200 --fetch-retries 1 --write-retries 1 --cpu-rounds 8000 --seed 7
python labs/python_concurrency/taskrun_round1_asyncio.py --help
python labs/python_concurrency/hybrid_three_stage_pipeline.py --help
```
- This lab is script-first, not harness-first.
- The right success metric is invariant safety plus bounded latency/throughput behavior, not one synthetic speedup scalar.

## Validation Checklist
- Each runnable scenario should preserve one terminal status per input item and deterministic ordered output where promised.
- Retry, cancellation, dedupe, and poison-path behavior should be visible in counters and summaries, not hidden.
- If this lab is later promoted into benchmark targets, add new explicit baseline/optimized files instead of retrofitting the current playbook scripts.

## Notes
- This is a control-plane lab, so the current documentation shape is intentionally different from the benchmark-pair labs.
