"""V1 EngineCore/CoreClient polling-loop demo (tool, not a benchmark).

This script illustrates the V1 guidance for polling an EngineCore:
- Keep polling even if `executed_flag` is false.
- Dedupe and report finished request IDs to enable KV-cache reclamation.
- Exit only when the CoreClient indicates all requests are complete.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterator, List, Set

from ch18.v1_engine_loop_common import MockRequestOutput, build_demo_stack  # noqa: E402


def baseline_engine_loop(engine_core: Any, core_client: Any) -> Iterator[MockRequestOutput]:
    """Naive polling loop that stops on an idle step.

    This is a pre-V1 style loop: it stops when EngineCore reports executed=False
    and yields no outputs, which can strand queued work and leave KV pages alive.
    """
    while True:
        outputs, executed = engine_core.step()
        for ro in outputs:
            yield ro

        finished_ids: List[str] = [ro.request_id for ro in outputs if getattr(ro, "finished", False)]
        if finished_ids:
            core_client.report_finished_ids(finished_ids)

        if not executed and not outputs:
            break


def run_engine_loop(engine_core: Any, core_client: Any) -> Iterator[MockRequestOutput]:
    """Poll the engine until the CoreClient reports all requests complete."""
    finished: Set[str] = set()

    while True:
        outputs, _executed = engine_core.step()

        for ro in outputs:
            yield ro

        newly_finished: Dict[str, MockRequestOutput] = {
            ro.request_id: ro for ro in outputs if getattr(ro, "finished", False) and ro.request_id not in finished
        }
        if newly_finished:
            core_client.report_finished_ids(list(newly_finished.keys()))
            finished.update(newly_finished.keys())

        if core_client.is_all_done():
            break


def _summarize(engine_core: Any, core_client: Any, outputs: List[MockRequestOutput]) -> Dict[str, object]:
    return {
        "steps": getattr(engine_core, "calls", None),
        "tokens": "".join(ro.delta_text for ro in outputs),
        "reported_finished": list(getattr(core_client, "finished_reported", [])),
        "all_done": core_client.is_all_done(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V1 EngineCore polling-loop demo (tool).")
    parser.add_argument(
        "--mode",
        choices=("baseline", "v1", "both"),
        default="both",
        help="Which loop to run (baseline=naive idle-break; v1=KV-safe).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode in ("baseline", "both"):
        engine_core, core_client = build_demo_stack()
        outputs = list(baseline_engine_loop(engine_core, core_client))
        print("Baseline loop demo:", _summarize(engine_core, core_client, outputs))

    if args.mode in ("v1", "both"):
        engine_core, core_client = build_demo_stack()
        outputs = list(run_engine_loop(engine_core, core_client))
        print("V1 loop demo:", _summarize(engine_core, core_client, outputs))


if __name__ == "__main__":
    main()
