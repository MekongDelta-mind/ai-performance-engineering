#!/usr/bin/env python3
"""Plot before/after heatmaps for a P2P bandwidth tool output.

This is intentionally small and robust: it parses the simple matrix printed
in the p2p-bandwidth output text files.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import apply_plot_style
import numpy as np


_ROW_RE = re.compile(r"^GPU\s+(\d+)\s+(.*)$")


def parse_matrix(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    # Find the printed matrix header line that starts with "GPU" and contains "0GPU".
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("GPU") and "0GPU" in line:
            header_idx = i
            break
    if header_idx is None:
        raise SystemExit(f"Could not find matrix header in {path}")

    rows = {}
    for line in lines[header_idx + 1 :]:
        m = _ROW_RE.match(line.strip())
        if not m:
            # Stop once the matrix rows end.
            if line.strip().startswith("Bandwidth Summary"):
                break
            continue
        idx = int(m.group(1))
        rest = m.group(2)
        tokens = rest.split()
        rows[idx] = tokens

    if not rows:
        raise SystemExit(f"No matrix rows parsed from {path}")

    n = max(rows.keys()) + 1
    mat = np.full((n, n), np.nan, dtype=np.float64)
    text = [["" for _ in range(n)] for _ in range(n)]

    for i in range(n):
        toks = rows.get(i)
        if toks is None:
            continue
        # Some outputs omit the final column spacing; tolerate short rows.
        for j in range(min(n, len(toks))):
            t = toks[j]
            if t == "-":
                mat[i, j] = np.nan
                text[i][j] = "-"
            elif t.upper() == "ERROR":
                mat[i, j] = np.nan
                text[i][j] = "ERR"
            else:
                try:
                    v = float(t)
                except ValueError:
                    mat[i, j] = np.nan
                    text[i][j] = t
                else:
                    mat[i, j] = v
                    text[i][j] = f"{v:.1f}"

    return mat, text


def plot(before_path: Path, after_path: Path, out_path: Path):
    before, before_txt = parse_matrix(before_path)
    after, after_txt = parse_matrix(after_path)

    vmax = 800.0  # NVLink-ish band for this platform; values above are clipped for color.

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mat, txt, title in (
        (axes[0], before, before_txt, "Before (invalid timing)"),
        (axes[1], after, after_txt, "After (fixed timing)"),
    ):
        shown = np.clip(mat, 0.0, vmax)
        im = ax.imshow(shown, vmin=0.0, vmax=vmax, cmap="viridis")
        ax.set_title(title)
        n = mat.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"GPU{j}" for j in range(n)])
        ax.set_yticklabels([f"GPU{i}" for i in range(n)])

        # Annotate values (including ERR/-).
        for i in range(n):
            for j in range(n):
                s = txt[i][j]
                if not s:
                    continue
                ax.text(j, i, s, ha="center", va="center", fontsize=8, color="white")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, label="GB/s (clipped at 800 for color)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    apply_plot_style()
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plot(Path(args.before), Path(args.after), Path(args.out))


if __name__ == "__main__":
    main()
