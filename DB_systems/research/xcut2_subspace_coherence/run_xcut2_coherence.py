#!/usr/bin/env python3
"""
Cross-cut X2 — Subspace coherence diagnostic (offline, no XQdrant change, no server).

Computes how well full-space k-NN sets agree with subspace k-NN sets across ratios. High
coherence => masking is "safe" (traversal in the subspace lands near the full-space answer);
low coherence predicts the recall collapse seen in Option 1. Correlate this curve with the
Option 1/2 recall curves to get a principled *why*.

Metric folder: experiments/<ts>__xcut2_subspace_coherence__subspace_coherence/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import TOP_K  # noqa: E402

STEP_ID = "xcut2_subspace_coherence"
METRIC = "subspace_coherence"
DEFAULT_RATIOS = [0.1, 0.25, 0.5, 0.75, 1.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dimensions", type=int, nargs="+", default=[768])
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--k", type=int, default=TOP_K)
    p.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    cr.add_dataset_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run = cr.init_research_run(
        STEP_ID, METRIC,
        mode="offline", dimensions=args.dimensions, queries=args.queries,
        requires_xqdrant_change="none (pure offline NumPy)",
        sweep={"ratios": args.ratios, "k": args.k},
        extra={"dataset": args.dataset},
    )
    print(f"[xcut2] experiment: {run.root}")

    for dataset in cr.iter_datasets(args):
        dim = dataset.dimension
        queries = dataset.queries[: min(args.queries, dataset.num_queries)]
        norms = cr.full_norms_sq(dataset)
        rows: list[dict] = []

        for ratio in args.ratios:
            dims = cr.subspace_indices(dim, ratio)
            subspace_gt = cr.SubspaceGT(dataset.vectors, dims, distance=dataset.distance)
            jaccards: list[float] = []
            overlaps: list[float] = []  # |intersection| / k  (== recall of subspace vs full)
            for i, q in enumerate(queries):
                full_ids = cr.full_top_k(dataset, i, q, args.k, norms)
                sub_ids, _ = subspace_gt.top_k(q, args.k)
                fs, ss = set(int(x) for x in full_ids), set(sub_ids.tolist())
                inter = len(fs & ss)
                union = len(fs | ss)
                jaccards.append(inter / union if union else 1.0)
                overlaps.append(inter / args.k)

            rows.append({
                "dimension": dim, "subspace_ratio": ratio, "subspace_dims": len(dims),
                "k": args.k,
                "coherence_overlap": float(np.mean(overlaps)),
                "coherence_jaccard": float(np.mean(jaccards)),
                "coherence_overlap_std": float(np.std(overlaps)),
            })
            print(f"  D={dim} ratio={ratio}: overlap={rows[-1]['coherence_overlap']:.3f} "
                  f"jaccard={rows[-1]['coherence_jaccard']:.3f}")

        csv_path = cr.bench_common.RESULTS_DIR / f"xcut2_coherence_d{dim}.csv"
        cr.write_csv(csv_path, list(rows[0].keys()), rows)
        cr.line_plot(
            [r["subspace_ratio"] for r in rows],
            {"Overlap (subspace∩full / k)": [r["coherence_overlap"] for r in rows],
             "Jaccard(subspace, full)": [r["coherence_jaccard"] for r in rows]},
            xlabel="Subspace ratio (D_sub / D)", ylabel="Coherence",
            title=f"Subspace coherence (D={dim})",
            stem=f"xcut2_coherence_d{dim}", hline=0.8,
        )

    print(f"[xcut2] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
