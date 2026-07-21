#!/usr/bin/env python3
"""
Cross-cut X2 — Subspace coherence diagnostic (offline, no XQdrant change, no server).

Computes how well full-space k-NN sets agree with subspace k-NN sets across ratios. High
coherence => masking is "safe" (traversal in the subspace lands near the full-space answer);
low coherence predicts the recall collapse seen in Option 1. Correlate this curve with the
Option 1/2 recall curves to get a principled *why*.

Emits mean±std across ``--trials`` independent query-sample seeds.

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
    cr.add_trials_arg(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.trials < 1:
        print("ERROR: --trials must be >= 1", file=sys.stderr)
        return 2

    run = cr.init_research_run(
        STEP_ID, METRIC,
        mode="offline", dimensions=args.dimensions, queries=args.queries,
        requires_xqdrant_change="none (pure offline NumPy)",
        sweep={"ratios": args.ratios, "k": args.k},
        extra={"dataset": args.dataset, "trials": args.trials},
    )
    print(f"[xcut2] experiment: {run.root}  trials={args.trials}")

    for dataset in cr.iter_datasets(args):
        dim = dataset.dimension
        norms = cr.full_norms_sq(dataset)
        trial_rows: list[dict] = []

        for trial in range(args.trials):
            seed = cr.trial_seed(trial)
            queries, q_idx = cr.select_queries(dataset, args.queries, seed=seed)

            for ratio in args.ratios:
                dims = cr.subspace_indices(dim, ratio, seed=seed)
                subspace_gt = cr.SubspaceGT(dataset.vectors, dims, distance=dataset.distance)
                jaccards: list[float] = []
                overlaps: list[float] = []  # |intersection| / k  (== recall of subspace vs full)
                for local_i, q in enumerate(queries):
                    src = int(q_idx[local_i])
                    full_ids = cr.full_top_k(dataset, src, q, args.k, norms)
                    sub_ids, _ = subspace_gt.top_k(q, args.k)
                    fs, ss = set(int(x) for x in full_ids), set(sub_ids.tolist())
                    inter = len(fs & ss)
                    union = len(fs | ss)
                    jaccards.append(inter / union if union else 1.0)
                    overlaps.append(inter / args.k)

                trial_rows.append({
                    "trial": trial,
                    "trial_seed": seed,
                    "dimension": dim,
                    "subspace_ratio": ratio,
                    "subspace_dims": len(dims),
                    "k": args.k,
                    "coherence_overlap": float(np.mean(overlaps)),
                    "coherence_jaccard": float(np.mean(jaccards)),
                })
                print(
                    f"  trial={trial} D={dim} ratio={ratio}: "
                    f"overlap={trial_rows[-1]['coherence_overlap']:.3f} "
                    f"jaccard={trial_rows[-1]['coherence_jaccard']:.3f}"
                )

        csv_path = cr.bench_common.RESULTS_DIR / f"xcut2_coherence_d{dim}.csv"
        rows = cr.write_trial_and_summary_csv(
            csv_path,
            trial_rows,
            key_fields=["dimension", "subspace_ratio", "k"],
        )
        cr.line_plot(
            [float(r["subspace_ratio"]) for r in rows],
            {
                "Overlap (subspace∩full / k)": [float(r["coherence_overlap"]) for r in rows],
                "Jaccard(subspace, full)": [float(r["coherence_jaccard"]) for r in rows],
            },
            yerr={
                "Overlap (subspace∩full / k)": [
                    float(r.get("coherence_overlap_std", 0.0) or 0.0) for r in rows
                ],
                "Jaccard(subspace, full)": [
                    float(r.get("coherence_jaccard_std", 0.0) or 0.0) for r in rows
                ],
            },
            xlabel="Subspace ratio (D_sub / D)",
            ylabel="Coherence (mean ± std)",
            title=f"C1: Subspace coherence (D={dim}, n={args.trials})",
            stem=f"xcut2_coherence_d{dim}",
            hline=0.8,
        )

    print(f"[xcut2] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
