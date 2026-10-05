#!/usr/bin/env python3
"""
Cross-cut X1 (paper: K1) — per-distance cost of masked kernels (plot side).

The measurement itself is the Rust bench ``masked_kernel_bench`` in the XQdrant fork (see
README). It compares three kernels per focus width ``d_sub``:

- ``repack``: copy focus dims into a scratch buffer, then the contiguous SIMD metric
- ``gather``: score focus dims in place (AVX2 ``vgatherdps`` on x86_64; scalar elsewhere)
- ``contiguous``: the same metric over ``d_sub`` already-contiguous coordinates; at
  ``d_sub == full_dim`` this is the plain full-vector distance

in two configurations (``wide``: Dot, full_dim 1536; ``sift``: Euclid, full_dim 128), and
emits a CSV::

    config,metric,full_dim,d_sub,kernel,ns_mean,ns_std,trials
    sift,euclid,128,32,gather,7.12,0.05,5
    ...

The legacy schema ``d_sub,kernel,ns_per_op`` (single configuration, no std) is still
accepted. One plot is written per configuration; the dashed line marks the full contiguous
distance when the CSV contains it.

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
FIELDS = ["config", "metric", "full_dim", "d_sub", "kernel", "ns_mean", "ns_std", "trials"]

EXAMPLE_ROWS = [
    {"config": "sift", "metric": "euclid", "full_dim": 128, "d_sub": d, "kernel": k,
     "ns_mean": ns, "ns_std": 0.0, "trials": 1}
    for d, k, ns in [
        (32, "repack", 9.0), (32, "gather", 7.0), (32, "contiguous", 3.0),
        (96, "repack", 22.0), (96, "gather", 16.0), (96, "contiguous", 6.0),
        (128, "repack", 26.0), (128, "gather", 20.0), (128, "contiguous", 7.0),
    ]
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel-csv", type=Path, default=None,
                   help="CSV emitted by the Rust bench (see module docstring)")
    p.add_argument("--make-example", action="store_true",
                   help="Write a template kernel_bench.csv (simulated) and plot it")
    return p.parse_args()


def normalize(rows: list[dict]) -> list[dict]:
    """Map legacy ``d_sub,kernel,ns_per_op`` rows onto the current schema."""
    out = []
    for r in rows:
        if "ns_mean" in r:
            out.append(r)
        else:
            out.append({"config": "wide", "metric": "dot", "full_dim": "", "d_sub": r["d_sub"],
                        "kernel": r["kernel"], "ns_mean": r["ns_per_op"], "ns_std": 0.0,
                        "trials": 1})
    return out


def plot_config(rows: list[dict], config: str, simulated: bool) -> None:
    rows = [r for r in rows if r["config"] == config]
    d_subs = sorted({int(r["d_sub"]) for r in rows})
    kernels = [k for k in ("repack", "gather", "contiguous") if any(r["kernel"] == k for r in rows)]
    series, yerr = {}, {}
    for kern in kernels:
        by_d = {int(r["d_sub"]): r for r in rows if r["kernel"] == kern}
        series[kern] = [float(by_d[d]["ns_mean"]) if d in by_d else float("nan") for d in d_subs]
        yerr[kern] = [float(by_d[d]["ns_std"]) if d in by_d else float("nan") for d in d_subs]

    metric = rows[0]["metric"]
    full_dim = str(rows[0]["full_dim"])
    full = [r for r in rows if r["kernel"] == "contiguous" and str(r["d_sub"]) == full_dim]
    hline = float(full[0]["ns_mean"]) if full else None
    trials = rows[0]["trials"]

    prefix = "[SIMULATED] " if simulated else ""
    width = f", D={full_dim}" if full_dim else ""
    cr.line_plot(
        d_subs, series,
        xlabel="D_sub (focus dimensions)",
        ylabel="ns per distance (lower = better)",
        title=f"{prefix}K1: masked kernel cost ({metric}{width}, T={trials})",
        stem=f"xcut1_kernel_{config}",
        hline=hline,
        yerr=yerr,
    )


def main() -> int:
    args = parse_args()
    run = cr.init_research_run(
        STEP_ID, METRIC, mode="offline", dimensions=[], queries=0,
        requires_xqdrant_change="Rust bench: gather / repack / contiguous masked kernels",
        sweep={"configs": ["wide", "sift"]},
    )
    print(f"[xcut1] experiment: {run.root}")

    dest_csv = cr.bench_common.RESULTS_DIR / "kernel_bench.csv"
    simulated = False
    if args.kernel_csv and args.kernel_csv.exists():
        rows = normalize(list(_csv.DictReader(args.kernel_csv.open(encoding="utf-8"))))
    elif args.make_example:
        rows = [dict(r) for r in EXAMPLE_ROWS]
        simulated = True
    else:
        print("  No --kernel-csv found. Run the Rust bench first, or pass --make-example.")
        print(f"  Expected schema: {','.join(FIELDS)}  -> {dest_csv}")
        return 1

    cr.write_csv(dest_csv, FIELDS, [{f: r.get(f, "") for f in FIELDS} for r in rows])
    for config in sorted({r["config"] for r in rows}):
        plot_config(rows, config, simulated)
    print(f"[xcut1] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
