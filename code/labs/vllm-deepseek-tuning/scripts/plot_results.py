#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_points(results_dir: Path):
    points = []
    for fp in sorted(results_dir.glob("*.json")):
        rec = json.loads(fp.read_text())
        m = rec.get("metrics", {})
        points.append(
            {
                "run_name": rec.get("run_name"),
                "scenario": rec.get("scenario"),
                "concurrency": rec.get("max_concurrency"),
                "returncode": rec.get("returncode"),
                "prefill": m.get("prefill_toks_per_s"),
                "decode": m.get("decode_toks_per_s"),
            }
        )
    return points


def plot_metric(points, scenario, metric, out_path: Path):
    grouped = {}
    for p in points:
        if p["scenario"] != scenario:
            continue
        if p["returncode"] != 0:
            continue
        if p[metric] is None:
            continue
        grouped.setdefault(p["run_name"], []).append((p["concurrency"], p[metric]))

    if not grouped:
        return False

    plt.figure(figsize=(10, 5))
    for run_name, vals in sorted(grouped.items()):
        vals = sorted(vals, key=lambda x: x[0])
        x = [v[0] for v in vals]
        y = [v[1] for v in vals]
        plt.plot(x, y, marker="o", linewidth=2, label=run_name)

    plt.title(f"{metric} throughput vs concurrency ({scenario})")
    plt.xlabel("max_concurrency")
    plt.ylabel("throughput (tok/s)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    points = load_points(results_dir)
    scenarios = sorted({p["scenario"] for p in points if p.get("scenario")})

    wrote = 0
    for sc in scenarios:
        if plot_metric(points, sc, "prefill", out_dir / f"{sc}_prefill.png"):
            wrote += 1
        if plot_metric(points, sc, "decode", out_dir / f"{sc}_decode.png"):
            wrote += 1

    print(f"Wrote {wrote} plot(s) to {out_dir}")


if __name__ == "__main__":
    main()
