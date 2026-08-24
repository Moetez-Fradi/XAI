#!/usr/bin/env python3
"""Fig 2 — cost model T_arith + T_io (lower bound with Delta T_mask = 0).

Instantiates arithmetic work from K1 ns/op ratios (gather vs repack vs full
contiguous SIMD). I/O is held constant so the plot shows why shrinking
arithmetic cannot yield an end-to-end win.

Paper numbers (K1 speedups vs repack): 1.24x / 1.14x / 1.42x / 1.77x at
D_sub in {32, 128, 384, 768}. Pass --kernel-csv to override.

Writes paper/figures/fig02_cost_model.png when --copy-paper-figures is set.
"""
from __future__ import annotations

import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402

STEP_ID = "cost_model"
METRIC = "t_arith_vs_t_io"

# Published K1 gather-vs-repack speedups (paper §5.7 / Fig 14).
DEFAULT_SPEEDUPS = {32: 1.24, 128: 1.14, 384: 1.42, 768: 1.77}
# Nominal full-D ns/op (Dot, contiguous) — relative units for the stacked bar.
FULL_NS = {32: 50.0, 128: 100.0, 384: 280.0, 768: 520.0}
# Graph I/O + heap/visited bookkeeping, independent of kernel (relative units).
T_IO = 900.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel-csv", type=Path, default=None)
    p.add_argument("--copy-paper-figures", action="store_true")
    return p.parse_args()


def _speedups_from_csv(path: Path) -> dict[int, float]:
    rows = list(_csv.DictReader(path.open(encoding="utf-8")))
    by: dict[int, dict[str, float]] = {}
    for r in rows:
        d = int(r["d_sub"])
        by.setdefault(d, {})[r["kernel"]] = float(r["ns_per_op"])
    out = {}
    for d, kern in by.items():
        if "gather" in kern and "repack" in kern and kern["gather"] > 0:
            out[d] = kern["repack"] / kern["gather"]
    return out


def main() -> int:
    args = parse_args()
    run = cr.init_research_run(
        STEP_ID, METRIC, mode="offline", dimensions=[], queries=0,
        requires_xqdrant_change="none (uses K1 ns/op)",
        sweep={"d_sub": list(DEFAULT_SPEEDUPS)},
    )
    speedups = dict(DEFAULT_SPEEDUPS)
    if args.kernel_csv and args.kernel_csv.exists():
        speedups.update(_speedups_from_csv(args.kernel_csv))

    d_subs = sorted(speedups)
    rows = []
    gather_arith = []
    repack_arith = []
    io_vals = []
    for d in d_subs:
        t_full = FULL_NS.get(d, FULL_NS[max(FULL_NS)])
        t_repack = t_full * 1.05  # scratch-buffer tax
        t_gather = t_repack / speedups[d]
        rows.append({
            "d_sub": d,
            "t_io": T_IO,
            "t_arith_full": t_full,
            "t_arith_repack": t_repack,
            "t_arith_gather": t_gather,
            "gather_vs_repack": speedups[d],
        })
        gather_arith.append(t_gather)
        repack_arith.append(t_repack)
        io_vals.append(T_IO)

    cr.write_csv(
        cr.bench_common.RESULTS_DIR / "cost_model.csv",
        list(rows[0].keys()), rows,
    )

    plt = cr._mpl()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(d_subs))
    w = 0.35
    ax.bar(x - w / 2, io_vals, w, label=r"$T_{\mathrm{io}}$", color="#4c72b0")
    ax.bar(x - w / 2, gather_arith, w, bottom=io_vals,
           label=r"$T_{\mathrm{arith}}$ gather", color="#55a868")
    ax.bar(x + w / 2, io_vals, w, color="#4c72b0", alpha=0.35)
    ax.bar(x + w / 2, repack_arith, w, bottom=io_vals,
           label=r"$T_{\mathrm{arith}}$ repack", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in d_subs])
    ax.set_xlabel(r"$D_{\mathrm{sub}}$")
    ax.set_ylabel("Relative cost (a.u.)")
    ax.set_title(r"Cost model: $T_{\mathrm{arith}}$ vs $T_{\mathrm{io}}$ ($\Delta T_{\mathrm{mask}}=0$)")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    png = cr.save_fig(fig, "fig02_cost_model")
    if args.copy_paper_figures:
        dest = cr.bench_common.PAPER_FIGURES_DIR / "fig02_cost_model.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(png.read_bytes())
    print(f"[cost_model] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
