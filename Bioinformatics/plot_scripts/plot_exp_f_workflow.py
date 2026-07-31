#!/usr/bin/env python3
"""Experiment F figure — workflow time comparison (Fig 5)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUTDIR, PALETTE, RESULTS, ensure_out, save, setup_style  # noqa: E402


def load_metrics_dir(sub: str) -> tuple[Path, dict, dict | None]:
    d = RESULTS / sub
    metrics_path = d / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text())
    indexed = None
    idx_path = d / "metrics_indexed.json"
    if idx_path.exists():
        indexed = json.loads(idx_path.read_text())
    return d, metrics, indexed


def main() -> int:
    setup_style()
    ensure_out()

    for sub in ("exp_f", "exp_f_smoke"):
        try:
            _, metrics, indexed = load_metrics_dir(sub)
            break
        except FileNotFoundError:
            continue
    else:
        print("ERROR: run Exp F first (./run_exp_f.sh)", file=sys.stderr)
        return 1

    pymol_s = float(metrics.get("pymol_manual_s", 600))
    xq = metrics["xqdrant"]
    fs = metrics["foldseek_tmalign_pymol"]
    bl = None

    labels = ["XQdrant\n(cold)", "Foldseek +\nTM-align"]
    auto = [xq["median_s"], fs["automated_only"]["median_s"]]
    manual = [0.0, pymol_s]
    colors_auto = [PALETTE["xqdrant"], PALETTE["foldseek"]]

    if indexed:
        xqi = indexed["xqdrant"]
        labels.insert(1, "XQdrant\n(indexed)")
        auto.insert(1, xqi["median_s"])
        manual.insert(1, 0.0)
        colors_auto.insert(1, PALETTE["esm2"])

    if not metrics["blast_tmalign_pymol"].get("skipped", True):
        bl = metrics["blast_tmalign_pymol"]
        labels.append("BLAST +\nTM-align")
        auto.append(bl["automated_only"]["median_s"])
        manual.append(pymol_s)
        colors_auto.append(PALETTE["blast"])

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(labels)), 4))
    x = np.arange(len(labels))
    w = 0.55
    ax.bar(x, auto, width=w, color=colors_auto, label="Automated")
    ax.bar(
        x,
        manual,
        width=w,
        bottom=auto,
        color="#94a3b8",
        hatch="//",
        edgecolor="white",
        label=f"Manual PyMOL (est. {pymol_s / 60:.0f} min)",
    )

    ymax = max(a + m for a, m in zip(auto, manual))
    for i, (a, m) in enumerate(zip(auto, manual)):
        total = a + m
        ax.text(
            i,
            total + max(ymax * 0.02, 1),
            f"{total:.2f}s" if total < 60 else f"{total:.0f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median wall-clock time per query (s)")
    ax.set_title("Experiment F — analyst workflow comparison")
    ax.legend(loc="upper left")

    steps = [
        f"{xq['automated_steps']} auto / {xq['manual_steps']} manual",
    ]
    if indexed:
        steps.append("1 auto / 0 manual")
    steps.append("2 auto / 1 manual")
    if bl is not None:
        steps.append("2 auto / 1 manual")
    for i, s in enumerate(steps):
        ax.text(i, 0.02 * ymax, s, ha="center", va="bottom", fontsize=8, color="#475569")

    save(fig, "exp_f_workflow_time")
    plt.close(fig)

    # Table figure
    fig2, ax2 = plt.subplots(figsize=(9, 2.6))
    ax2.axis("off")
    rows = [
        ["Workflow", "Auto steps", "Manual", "Median auto (s)", "Median total (s)", "Rationale"],
        [
            "XQdrant (cold)",
            str(xq["automated_steps"]),
            "0",
            f"{xq['median_s']:.2f}",
            f"{xq['median_s']:.2f}",
            "embed + dims_explained",
        ],
    ]
    if indexed:
        rows.append(
            [
                "XQdrant (indexed)",
                "1",
                "0",
                f"{indexed['xqdrant']['median_s']:.3f}",
                f"{indexed['xqdrant']['median_s']:.3f}",
                "search + dims_explained",
            ]
        )
    rows.append(
        [
            "Foldseek→TM→PyMOL",
            "2",
            "1",
            f"{fs['automated_only']['median_s']:.2f}",
            f"{fs['with_manual_pymol']['median_s']:.0f}",
            "visual inspection",
        ]
    )
    if bl is not None:
        rows.append(
            [
                "BLAST→TM→PyMOL",
                "2",
                "1",
                f"{bl['automated_only']['median_s']:.2f}",
                f"{bl['with_manual_pymol']['median_s']:.0f}",
                "visual inspection",
            ]
        )
    table = ax2.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for j in range(len(rows[0])):
        table[(0, j)].set_facecolor("#e2e8f0")
    save(fig2, "exp_f_workflow_table")
    plt.close(fig2)

    print(f"Wrote {OUTDIR}/exp_f_workflow_time.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
