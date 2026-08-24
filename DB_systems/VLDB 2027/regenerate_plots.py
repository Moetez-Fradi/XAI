#!/usr/bin/env python3
"""Regenerate publication charts from existing CSV results (no re-benchmark)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import bench_common
from bench_common import ensure_output_dirs, use_experiment_dir
from bench_viz import (
    plot_attribution_depth,
    plot_latency_recall,
    plot_masked_subspace,
    plot_subspace_speedup,
    plot_throughput,
)


PLOTTERS = {
    "1": plot_latency_recall,
    "2": plot_throughput,
    "3": plot_attribution_depth,
    "4": plot_subspace_speedup,
    "5": plot_masked_subspace,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate plots from CSV results")
    parser.add_argument(
        "--experiment",
        type=Path,
        help="Experiment folder (experiments/<timestamp>) or its results/ subdir",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[int(x) for x in os.environ.get("BENCH_DIMENSIONS", "768 1536").split()],
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["1", "2", "3", "4", "5", "all"],
        default=["all"],
        help="Which plots to regenerate (default: all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.experiment:
        run = use_experiment_dir(args.experiment)
        print(f"Using experiment: {run.root}")
    else:
        ensure_output_dirs()
        print(f"Using results dir: {bench_common.RESULTS_DIR}")

    plot_ids = list(PLOTTERS) if "all" in args.plots else args.plots
    created = []

    for dim in args.dimensions:
        for plot_id in plot_ids:
            path = PLOTTERS[plot_id](dim)
            if path is not None:
                created.append(path)
                print(f"  wrote {path}")

    if not created:
        print(f"No plots generated. Check CSV files in {bench_common.RESULTS_DIR}", file=sys.stderr)
        return 1

    print(f"\nDone. Charts in {bench_common.PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
