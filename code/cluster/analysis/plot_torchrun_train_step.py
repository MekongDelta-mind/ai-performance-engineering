#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


def main() -> int:
    apply_plot_style()
    p = argparse.ArgumentParser(description="Plot per-step times from torchrun train-step benchmark.")
    p.add_argument("--input", required=True, help="Structured JSON from run_torchrun_transformer_train_step.sh")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--title", default="")
    args = p.parse_args()

    d = json.loads(Path(args.input).read_text(encoding="utf-8"))
    step_times = (d.get("results") or {}).get("step_times_s") or []
    if not step_times:
        raise SystemExit("No step_times_s in input JSON.")

    xs = list(range(1, len(step_times) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, step_times, marker="o", linewidth=1.2)
    ax.set_xlabel("Measured step")
    ax.set_ylabel("Step time (s) [max across ranks]")

    title = args.title or f"Train Step Times ({d.get('label','')}, world={d.get('world_size','')})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

