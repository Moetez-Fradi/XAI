#!/usr/bin/env python3
"""Pipeline schematic for PSB paper (Fig 6)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out, save, setup_style  # noqa: E402


def box(ax, xy, w, h, text, fc="#dbeafe", ec="#2563eb"):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=ec, lw=1.5
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5),
    )


def main() -> int:
    setup_style()
    ensure_out()
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    y = 1.0
    h = 0.9
    w = 1.35
    xs = [0.2, 1.8, 3.4, 5.0, 6.6, 8.2]
    labels = [
        "PDB\nchain",
        "ESM2\nembed",
        "XQdrant\nindex",
        "Query +\nretrieval",
        "dims_\nexplained",
        "DSSP map\n+ PyMOL",
    ]
    colors = ["#e2e8f0", "#dbeafe", "#dbeafe", "#bfdbfe", "#93c5fd", "#fef3c7"]
    edges = ["#64748b", "#2563eb", "#2563eb", "#2563eb", "#1d4ed8", "#d97706"]

    for i, (x, lab, fc, ec) in enumerate(zip(xs, labels, colors, edges)):
        box(ax, (x, y), w, h, lab, fc=fc, ec=ec)
        if i < len(xs) - 1:
            arrow(ax, x + w + 0.05, y + h / 2, xs[i + 1] - 0.05, y + h / 2)

    ax.text(
        5.0,
        2.55,
        "XQdrant explainable protein retrieval pipeline",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(5.0, 0.25, "Train-only dim→profile map; holdout locked before tuning", ha="center", fontsize=8, color="#64748b")

    save(fig, "pipeline_schematic")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
