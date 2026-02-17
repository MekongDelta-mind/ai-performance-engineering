#!/usr/bin/env python3
"""Shared plotting style for cluster evaluation figures."""

from __future__ import annotations

import matplotlib as mpl

COLOR_CYCLE = [
    "#1F77B4",  # blue
    "#FF7F0E",  # orange
    "#2CA02C",  # green
    "#D62728",  # red
    "#9467BD",  # purple
    "#8C564B",  # brown
]


def apply_plot_style() -> None:
    """Apply a consistent visual style for all generated report figures."""
    mpl.rcParams.update(
        {
            "axes.grid": True,
            "axes.labelsize": 10,
            "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titlesize": 12,
            "font.size": 10,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "legend.fontsize": 9,
            "savefig.dpi": 220,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
