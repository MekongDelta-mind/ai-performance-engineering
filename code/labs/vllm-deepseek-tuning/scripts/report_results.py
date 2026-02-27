#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load_records(results_dir: Path):
    rows = []
    for fp in sorted(results_dir.glob("*.json")):
        rec = json.loads(fp.read_text())
        m = rec.get("metrics", {})
        rows.append(
            {
                "run_name": rec.get("run_name"),
                "model": rec.get("model"),
                "scenario": rec.get("scenario"),
                "isl": rec.get("isl"),
                "osl": rec.get("osl"),
                "concurrency": rec.get("max_concurrency"),
                "returncode": rec.get("returncode"),
                "prefill_toks_per_s": m.get("prefill_toks_per_s"),
                "decode_toks_per_s": m.get("decode_toks_per_s"),
                "raw_file": rec.get("raw_file"),
            }
        )
    return rows


def best_by(rows, scenario, run_name, metric):
    vals = [r for r in rows if r["scenario"] == scenario and r["run_name"] == run_name and r.get(metric) is not None and r.get("returncode") == 0]
    if not vals:
        return None
    return max(vals, key=lambda x: x[metric])


def mean_metric(rows, scenario, run_name, metric):
    vals = [r[metric] for r in rows if r["scenario"] == scenario and r["run_name"] == run_name and r.get(metric) is not None and r.get("returncode") == 0]
    return mean(vals) if vals else None


def pct_gain(a, b):
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b * 100.0


def fmt(v, nd=1):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def write_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_name",
        "model",
        "scenario",
        "isl",
        "osl",
        "concurrency",
        "returncode",
        "prefill_toks_per_s",
        "decode_toks_per_s",
        "raw_file",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows, out_md: Path):
    out_md.parent.mkdir(parents=True, exist_ok=True)

    runs = sorted({r["run_name"] for r in rows})
    scenarios = sorted({r["scenario"] for r in rows})

    # best rows per run/scenario
    best_map = defaultdict(dict)
    for run in runs:
        for sc in scenarios:
            bp = best_by(rows, sc, run, "prefill_toks_per_s")
            bd = best_by(rows, sc, run, "decode_toks_per_s")
            best_map[run][sc] = {"prefill": bp, "decode": bd}

    # blog-like comparisons
    v32_tp2_prefill = best_by(rows, "prefill_only", "ds_v32_nvfp4_tp2", "prefill_toks_per_s")
    v32_tp4_prefill = best_by(rows, "prefill_only", "ds_v32_nvfp4_tp4", "prefill_toks_per_s")
    v32_tp2_mix1k = best_by(rows, "mixed_moderate_1k", "ds_v32_nvfp4_tp2", "decode_toks_per_s")
    v32_tp4_mix1k = best_by(rows, "mixed_moderate_1k", "ds_v32_nvfp4_tp4", "decode_toks_per_s")

    r1_tp2_prefill = best_by(rows, "prefill_only", "ds_r1_nvfp4_tp2", "prefill_toks_per_s")
    r1_ep2_prefill = best_by(rows, "prefill_only", "ds_r1_nvfp4_ep2", "prefill_toks_per_s")
    r1_tp2_mix64 = best_by(rows, "mixed_short_64", "ds_r1_nvfp4_tp2", "decode_toks_per_s")
    r1_ep2_mix64 = best_by(rows, "mixed_short_64", "ds_r1_nvfp4_ep2", "decode_toks_per_s")

    r1_tp2_mtp1_mix64 = best_by(rows, "mixed_short_64", "ds_r1_nvfp4_tp2_mtp1", "decode_toks_per_s")

    r1_vs_v32_prefill_ratio = None
    if r1_tp2_prefill and v32_tp2_prefill and v32_tp2_prefill["prefill_toks_per_s"]:
        r1_vs_v32_prefill_ratio = r1_tp2_prefill["prefill_toks_per_s"] / v32_tp2_prefill["prefill_toks_per_s"]

    lines = []
    lines.append("# vLLM DeepSeek Tuning Report")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Records analyzed: **{len(rows)}**")
    lines.append(f"- Successful records: **{sum(1 for r in rows if r['returncode'] == 0)}**")
    lines.append(f"- Scenarios: {', '.join(scenarios)}")
    lines.append("")
    lines.append("## Blog-aligned comparisons")
    lines.append("")

    lines.append("### DeepSeek-V3.2 TP2 vs TP4")
    lines.append(f"- Prefill-only best (TP2): {fmt(v32_tp2_prefill['prefill_toks_per_s']) if v32_tp2_prefill else 'n/a'} tok/s @ c={v32_tp2_prefill['concurrency'] if v32_tp2_prefill else 'n/a'}")
    lines.append(f"- Prefill-only best (TP4): {fmt(v32_tp4_prefill['prefill_toks_per_s']) if v32_tp4_prefill else 'n/a'} tok/s @ c={v32_tp4_prefill['concurrency'] if v32_tp4_prefill else 'n/a'}")
    if v32_tp2_prefill and v32_tp4_prefill:
        lines.append(f"- Prefill TP2 vs TP4 gain: {fmt(pct_gain(v32_tp2_prefill['prefill_toks_per_s'], v32_tp4_prefill['prefill_toks_per_s']))}%")

    lines.append(f"- Mixed (ISL=2k, OSL=1k) best decode (TP2): {fmt(v32_tp2_mix1k['decode_toks_per_s']) if v32_tp2_mix1k else 'n/a'} tok/s")
    lines.append(f"- Mixed (ISL=2k, OSL=1k) best decode (TP4): {fmt(v32_tp4_mix1k['decode_toks_per_s']) if v32_tp4_mix1k else 'n/a'} tok/s")
    if v32_tp2_mix1k and v32_tp4_mix1k:
        lines.append(f"- Mixed decode TP2 vs TP4 gain: {fmt(pct_gain(v32_tp2_mix1k['decode_toks_per_s'], v32_tp4_mix1k['decode_toks_per_s']))}%")
    lines.append("")

    lines.append("### DeepSeek-R1 TP2 vs EP2")
    lines.append(f"- Prefill-only best (TP2): {fmt(r1_tp2_prefill['prefill_toks_per_s']) if r1_tp2_prefill else 'n/a'} tok/s")
    lines.append(f"- Prefill-only best (EP2): {fmt(r1_ep2_prefill['prefill_toks_per_s']) if r1_ep2_prefill else 'n/a'} tok/s")
    if r1_tp2_prefill and r1_ep2_prefill:
        lines.append(f"- Prefill EP2 vs TP2 gain: {fmt(pct_gain(r1_ep2_prefill['prefill_toks_per_s'], r1_tp2_prefill['prefill_toks_per_s']))}%")

    lines.append(f"- Mixed short (OSL=64) best decode (TP2): {fmt(r1_tp2_mix64['decode_toks_per_s']) if r1_tp2_mix64 else 'n/a'} tok/s")
    lines.append(f"- Mixed short (OSL=64) best decode (EP2): {fmt(r1_ep2_mix64['decode_toks_per_s']) if r1_ep2_mix64 else 'n/a'} tok/s")
    if r1_tp2_mix64 and r1_ep2_mix64:
        lines.append(f"- Mixed decode TP2 vs EP2 gain: {fmt(pct_gain(r1_tp2_mix64['decode_toks_per_s'], r1_ep2_mix64['decode_toks_per_s']))}%")
    lines.append("")

    lines.append("### MTP effect (R1, OSL=64)")
    lines.append(f"- Best decode without MTP (TP2): {fmt(r1_tp2_mix64['decode_toks_per_s']) if r1_tp2_mix64 else 'n/a'} tok/s")
    lines.append(f"- Best decode with MTP=1 (TP2): {fmt(r1_tp2_mtp1_mix64['decode_toks_per_s']) if r1_tp2_mtp1_mix64 else 'n/a'} tok/s")
    if r1_tp2_mix64 and r1_tp2_mtp1_mix64:
        lines.append(f"- MTP gain/loss: {fmt(pct_gain(r1_tp2_mtp1_mix64['decode_toks_per_s'], r1_tp2_mix64['decode_toks_per_s']))}%")
    lines.append("")

    lines.append("### DeepSeek-R1 vs DeepSeek-V3.2 (Prefill-only, TP2)")
    lines.append(f"- Throughput ratio (R1/V3.2): {fmt(r1_vs_v32_prefill_ratio, 2)}x")
    lines.append("")

    lines.append("## Per-run scenario bests")
    lines.append("")
    for run in runs:
        lines.append(f"### {run}")
        for sc in scenarios:
            bp = best_map[run][sc]["prefill"]
            bd = best_map[run][sc]["decode"]
            lines.append(
                f"- {sc}: prefill={fmt(bp['prefill_toks_per_s']) if bp else 'n/a'} tok/s @ c={bp['concurrency'] if bp else 'n/a'}; "
                f"decode={fmt(bd['decode_toks_per_s']) if bd else 'n/a'} tok/s @ c={bd['concurrency'] if bd else 'n/a'}"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("- Absolute numbers are hardware-dependent. Use relative deltas for cross-platform interpretation.")
    lines.append("- Failed records are retained for auditability; only successful runs are used for best-of comparisons.")

    out_md.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_records(results_dir)
    write_csv(rows, out_dir / "summary.csv")
    write_markdown(rows, out_dir / "report.md")

    print(f"Wrote report artifacts to {out_dir}")


if __name__ == "__main__":
    main()
