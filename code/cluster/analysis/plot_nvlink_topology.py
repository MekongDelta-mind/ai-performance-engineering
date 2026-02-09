#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
GPU_RE = re.compile(r"^GPU\d+$")
NV_RE = re.compile(r"^NV(\d+)$")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def read_topology_stdout(meta_path: Path) -> tuple[str, str, str]:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    label = str(payload.get("label") or meta_path.stem)
    commands = payload.get("commands") or {}
    topo = (commands.get("nvidia_smi_topo") or {}).get("stdout")
    if not topo:
        raise ValueError(f"No commands.nvidia_smi_topo.stdout in {meta_path}")

    run_id = "unknown"
    stem = meta_path.stem
    if stem.endswith("_meta"):
        run_id = stem[: -len("_meta")]
    return label, run_id, str(topo)


def parse_gpu_matrix(stdout: str) -> tuple[list[str], dict[str, list[str]]]:
    lines = [strip_ansi(line).rstrip("\n") for line in stdout.splitlines()]
    lines = [line for line in lines if line.strip()]

    header_tokens: list[str] | None = None
    header_idx = -1
    for idx, line in enumerate(lines):
        toks = [tok.strip() for tok in line.split("\t") if tok.strip()]
        if toks and all(GPU_RE.match(tok) for tok in toks[: min(2, len(toks))]):
            header_tokens = toks
            header_idx = idx
            break
    if not header_tokens:
        raise ValueError("Unable to locate GPU header in nvidia-smi topo output")

    gpu_labels: list[str] = []
    for tok in header_tokens:
        if GPU_RE.match(tok):
            gpu_labels.append(tok)
        elif gpu_labels:
            break
    if not gpu_labels:
        raise ValueError("No GPU columns detected in topo output")

    matrix: dict[str, list[str]] = {}
    for line in lines[header_idx + 1 :]:
        toks = [tok.strip() for tok in line.split("\t")]
        if not toks:
            continue
        if line.strip().startswith("Legend:"):
            break

        row_name = toks[0].strip()
        if not GPU_RE.match(row_name):
            continue

        row_vals: list[str] = []
        for i in range(len(gpu_labels)):
            idx = i + 1
            row_vals.append(toks[idx].strip() if idx < len(toks) else "")
        matrix[row_name] = row_vals

    if len(matrix) != len(gpu_labels):
        missing = [g for g in gpu_labels if g not in matrix]
        raise ValueError(f"Missing GPU rows in topo output: {missing}")

    return gpu_labels, matrix


def link_score(link: str) -> float:
    v = (link or "").strip().upper()
    if v == "X":
        return 0.0
    m = NV_RE.match(v)
    if m:
        return 200.0 + float(m.group(1))

    mapping = {
        "PIX": 60.0,
        "PXB": 50.0,
        "PHB": 40.0,
        "NODE": 30.0,
        "SYS": 20.0,
    }
    return mapping.get(v, 10.0)


def build_summary(gpu_labels: list[str], matrix: dict[str, list[str]]) -> dict:
    pair_counts: dict[str, int] = {}
    nv_widths: list[int] = []

    for i, src in enumerate(gpu_labels):
        row = matrix[src]
        for j in range(i + 1, len(gpu_labels)):
            link = (row[j] or "").strip()
            key = link if link else "UNKNOWN"
            pair_counts[key] = pair_counts.get(key, 0) + 1
            m = NV_RE.match(key.upper())
            if m:
                nv_widths.append(int(m.group(1)))

    summary = {
        "gpu_count": len(gpu_labels),
        "gpu_labels": gpu_labels,
        "link_pair_counts": dict(sorted(pair_counts.items(), key=lambda kv: kv[0])),
        "nvlink_pair_count": len(nv_widths),
        "nvlink_width_min": min(nv_widths) if nv_widths else None,
        "nvlink_width_max": max(nv_widths) if nv_widths else None,
        "nvlink_width_mean": (sum(nv_widths) / len(nv_widths)) if nv_widths else None,
        "gpu_matrix": {
            src: {dst: matrix[src][j] for j, dst in enumerate(gpu_labels)} for src in gpu_labels
        },
    }
    return summary


def render_figure(
    gpu_labels: list[str],
    matrix: dict[str, list[str]],
    out_path: Path,
    title: str,
) -> None:
    n = len(gpu_labels)
    scores = np.zeros((n, n), dtype=float)
    labels = [["" for _ in range(n)] for _ in range(n)]

    for i, src in enumerate(gpu_labels):
        for j, dst in enumerate(gpu_labels):
            link = (matrix[src][j] or "").strip()
            labels[i][j] = link
            scores[i, j] = link_score(link)

    fig, ax = plt.subplots(figsize=(max(7.5, 1.25 * n + 3.0), max(6.0, 1.05 * n + 2.5)))
    im = ax.imshow(scores, cmap="YlGnBu")

    ax.set_xticks(range(n), gpu_labels, rotation=0)
    ax.set_yticks(range(n), gpu_labels)
    ax.set_xlabel("Destination GPU")
    ax.set_ylabel("Source GPU")
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            text_color = "black"
            if scores[i, j] >= 170:
                text_color = "white"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=9, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative link strength (higher is closer/faster)")

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot NVLink/NVSwitch GPU topology from node meta nvidia-smi topo output."
    )
    parser.add_argument("--meta", required=True, help="Path to results/structured/<run_id>_<label>_meta.json")
    parser.add_argument("--fig-out", required=True, help="Output PNG path")
    parser.add_argument("--summary-out", default="", help="Optional output JSON summary path")
    parser.add_argument("--title", default="", help="Optional chart title override")
    args = parser.parse_args()

    apply_plot_style()

    meta_path = Path(args.meta)
    fig_out = Path(args.fig_out)

    label, run_ref, stdout = read_topology_stdout(meta_path)
    gpu_labels, matrix = parse_gpu_matrix(stdout)

    title = args.title or f"GPU NVLink/NVSwitch Topology ({label})"
    render_figure(gpu_labels, matrix, fig_out, title)

    if args.summary_out:
        summary = build_summary(gpu_labels, matrix)
        summary_payload = {
            "meta_path": str(meta_path),
            "label": label,
            "run_ref": run_ref,
            "figure": str(fig_out),
            "summary": summary,
        }
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
