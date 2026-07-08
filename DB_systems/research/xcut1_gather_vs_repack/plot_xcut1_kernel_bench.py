#!/usr/bin/env python3
"""
Cross-cut X1 — SIMD gather vs. repack kernel benchmark (plot side).

The measurement itself is a Rust ``criterion`` benchmark (see README) comparing gather-based
kernels vs. a repack-into-scratch-buffer at D_sub in {32, 128, 384}. That bench must emit a CSV:

    d_sub,kernel,ns_per_op
    32,gather,41.2
    32,repack,33.8
    128,gather,88.0
    ...

This script reads that CSV and plots ns/op (lower = better) per kernel across D_sub. Pass
``--make-example`` to drop a template CSV so you can validate the plot before the Rust bench exists.

Metric folder: experiments/<ts>__xcut1_gather_vs_repack__kernel_gather_vs_repack/
"""

from __future__ import annotations

import argparse
import csv as _csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402

STEP_ID = "xcut1_gather_vs_repack"
METRIC = "kernel_gather_vs_repack"

EXAMPLE_ROWS = [
    {"d_sub": 32, "kernel": "gather", "ns_per_op": 41.2},
    {"d_sub": 32, "kernel": "repack", "ns_per_op": 33.8},
    {"d_sub": 128, "kernel": "gather", "ns_per_op": 88.0},
    {"d_sub": 128, "kernel": "repack", "ns_per_op": 92.5},
    {"d_sub": 384, "kernel": "gather", "ns_per_op": 210.0},
    {"d_sub": 384, "kernel": "repack", "ns_per_op": 240.0},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel-csv", type=Path, default=None,
                   help="CSV emitted by the Rust criterion bench (d_sub,kernel,ns_per_op)")
    p.add_argument("--make-example", action="store_true",
                   help="Write a template kernel_bench.csv (simulated) and plot it")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run = cr.init_research_run(
        STEP_ID, METRIC, mode="offline", dimensions=[], queries=0,
        requires_xqdrant_change="Rust criterion bench: gather kernels vs repack scratch buffer",
        sweep={"d_sub": [32, 128, 384]},
    )
    print(f"[xcut1] experiment: {run.root}")

    dest_csv = cr.bench_common.RESULTS_DIR / "kernel_bench.csv"
    simulated = False
    if args.kernel_csv and args.kernel_csv.exists():
        rows = list(_csv.DictReader(args.kernel_csv.open(encoding="utf-8")))
    elif args.make_example:
        rows = [dict(r) for r in EXAMPLE_ROWS]
        simulated = True
    else:
        print("  No --kernel-csv found. Run the Rust bench first, or pass --make-example.")
        print(f"  Expected schema: d_sub,kernel,ns_per_op  -> {dest_csv}")
        return 1

    cr.write_csv(dest_csv, ["d_sub", "kernel", "ns_per_op"],
                 [{"d_sub": r["d_sub"], "kernel": r["kernel"], "ns_per_op": r["ns_per_op"]}
                  for r in rows])

    d_subs = sorted({int(r["d_sub"]) for r in rows})
    kernels = sorted({r["kernel"] for r in rows})
    series = {}
    for kern in kernels:
        by_d = {int(r["d_sub"]): float(r["ns_per_op"]) for r in rows if r["kernel"] == kern}
        series[kern] = [by_d.get(d, float("nan")) for d in d_subs]

    prefix = "[SIMULATED] " if simulated else ""
    cr.line_plot(
        d_subs, series,
        xlabel="D_sub (focus dimensions)", ylabel="ns per distance op (lower = better)",
        title=f"{prefix}Gather vs. repack kernel",
        stem="xcut1_kernel_gather_vs_repack",
    )
    print(f"[xcut1] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
