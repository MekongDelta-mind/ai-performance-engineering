#!/usr/bin/env python3
"""
Plot key metrics across repeated cluster health suite runs (base vs extended).

Inputs are the suite summary JSONs written by scripts/run_cluster_health_suite.sh.
This intentionally keeps dependencies minimal: matplotlib + stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _detect_mode(summary: Dict[str, Any]) -> str:
    # "extended" runs include extra IB and/or NCCL alltoall.
    if "ib_read_bw" in summary or "ib_send_bw" in summary:
        return "extended"
    nccl = summary.get("nccl") or {}
    if isinstance(nccl, dict) and "alltoall_perf" in nccl:
        return "extended"
    return "base"


_RE_REPEAT = re.compile(r"_r(?P<rep>[0-9]+)_(?P<mode>base|extended)\b")


def _repeat_index(run_id: str) -> Optional[int]:
    m = _RE_REPEAT.search(run_id)
    if not m:
        return None
    try:
        return int(m.group("rep"))
    except ValueError:
        return None


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _ib_mean(summary: Dict[str, Any], key: str) -> Optional[float]:
    ib = summary.get(key)
    if not isinstance(ib, dict) or not ib:
        return None
    vals = []
    for payload in ib.values():
        if not isinstance(payload, dict):
            continue
        v = payload.get("avg_gbps")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return _mean(vals)


def _nccl_max(summary: Dict[str, Any], coll: str) -> Optional[float]:
    nccl = summary.get("nccl") or {}
    if not isinstance(nccl, dict):
        return None
    payload = nccl.get(coll) or {}
    if not isinstance(payload, dict):
        return None
    mb = payload.get("max_busbw") or {}
    if not isinstance(mb, dict):
        return None
    v = mb.get("busbw_gbps")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _torchdist_max(summary: Dict[str, Any]) -> Optional[float]:
    td = summary.get("torchdist") or {}
    if not isinstance(td, dict):
        return None
    mb = td.get("max_busbw") or {}
    if not isinstance(mb, dict):
        return None
    v = mb.get("busbw_gbps")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _iperf(summary: Dict[str, Any], direction: str) -> Optional[float]:
    ip = summary.get("iperf3") or {}
    if not isinstance(ip, dict):
        return None
    d = ip.get(direction) or {}
    if not isinstance(d, dict):
        return None
    v = d.get("gbps")
    if isinstance(v, (int, float)):
        return float(v)
    return None


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    units: str
    getter: Callable[[Dict[str, Any]], Optional[float]]


METRICS: List[Metric] = [
    Metric("iperf3.fwd", "TCP iperf3 fwd", "Gbps", lambda s: _iperf(s, "fwd")),
    Metric("iperf3.rev", "TCP iperf3 rev", "Gbps", lambda s: _iperf(s, "rev")),
    Metric("ib_write_bw.mean", "IB write bw (mean over HCAs)", "Gbps", lambda s: _ib_mean(s, "ib_write_bw")),
    Metric("ib_read_bw.mean", "IB read bw (mean over HCAs)", "Gbps", lambda s: _ib_mean(s, "ib_read_bw")),
    Metric("ib_send_bw.mean", "IB send bw (mean over HCAs)", "Gbps", lambda s: _ib_mean(s, "ib_send_bw")),
    Metric("nccl.all_reduce_perf.max", "NCCL all_reduce max busbw", "GB/s", lambda s: _nccl_max(s, "all_reduce_perf")),
    Metric("nccl.all_gather_perf.max", "NCCL all_gather max busbw", "GB/s", lambda s: _nccl_max(s, "all_gather_perf")),
    Metric(
        "nccl.reduce_scatter_perf.max",
        "NCCL reduce_scatter max busbw",
        "GB/s",
        lambda s: _nccl_max(s, "reduce_scatter_perf"),
    ),
    Metric("nccl.alltoall_perf.max", "NCCL alltoall max busbw", "GB/s", lambda s: _nccl_max(s, "alltoall_perf")),
    Metric("torchdist.max", "Torchdist all_reduce max busbw", "GB/s", _torchdist_max),
]


def _load_summaries(paths: List[Path]) -> List[Tuple[Path, Dict[str, Any]]]:
    out = []
    for p in paths:
        try:
            out.append((p, _read_json(p)))
        except Exception as exc:
            raise SystemExit(f"ERROR: failed to read {p}: {exc}")
    return out


def _series(
    summaries: List[Tuple[Path, Dict[str, Any]]], metric: Metric
) -> Dict[str, List[Tuple[int, float, str]]]:
    # Return {mode: [(rep, value, run_id), ...]}
    out: Dict[str, List[Tuple[int, float, str]]] = {"base": [], "extended": []}
    for _path, s in summaries:
        run_id = str(s.get("run_id") or "")
        mode = _detect_mode(s)
        rep = _repeat_index(run_id)
        if rep is None:
            # Fall back to stable ordering by run_id; use a synthetic index.
            # (This should not happen for the repeats runner.)
            continue
        v = metric.getter(s)
        if v is None:
            continue
        out.setdefault(mode, []).append((rep, float(v), run_id))
    for m in out:
        out[m].sort(key=lambda t: t[0])
    return out


def _all_repeat_indices(summaries: List[Tuple[Path, Dict[str, Any]]]) -> List[int]:
    reps: List[int] = []
    for _path, s in summaries:
        rid = str(s.get("run_id") or "")
        rep = _repeat_index(rid)
        if rep is not None:
            reps.append(rep)
    return sorted(set(reps))


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot repeated cluster health suite metrics.")
    ap.add_argument("--glob", required=True, help="Glob for *_cluster_health_suite_summary.json files")
    ap.add_argument("--output", required=True, help="Output figure path (PNG recommended)")
    ap.add_argument("--title", default="", help="Optional title")
    args = ap.parse_args()

    paths = sorted(Path(".").glob(args.glob))
    if not paths:
        raise SystemExit(f"ERROR: no files matched --glob {args.glob!r}")

    summaries = _load_summaries(paths)
    reps = _all_repeat_indices(summaries)
    if not reps:
        raise SystemExit("ERROR: could not parse repeat indices from run_id fields")

    # 4x3 grid = 12 slots; we currently have <=10 metrics.
    ncols = 3
    nrows = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5.2, nrows * 3.2))
    axes_list = [ax for row in axes for ax in row]

    handles_labels = None
    for i, metric in enumerate(METRICS):
        if i >= len(axes_list):
            break
        ax = axes_list[i]
        series = _series(summaries, metric)

        plotted_any = False
        for mode, marker, color in (("base", "o", "#1f77b4"), ("extended", "s", "#ff7f0e")):
            pts = series.get(mode) or []
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker=marker, linestyle="-", linewidth=1.5, color=color, label=mode)
            plotted_any = True

        if not plotted_any:
            ax.axis("off")
            continue

        ax.set_title(metric.title, fontsize=10)
        ax.set_xlabel("Repeat")
        ax.set_ylabel(metric.units)
        ax.set_xticks(reps)
        ax.grid(True, linestyle="--", alpha=0.35)

        if handles_labels is None:
            cand = ax.get_legend_handles_labels()
            if cand[0]:
                handles_labels = cand

    # Hide any unused axes.
    for j in range(len(METRICS), len(axes_list)):
        axes_list[j].axis("off")

    if handles_labels is not None:
        handles, labels = handles_labels
        if handles:
            fig.legend(handles, labels, loc="upper right", frameon=False)

    if args.title:
        fig.suptitle(args.title, y=0.995, fontsize=12)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
