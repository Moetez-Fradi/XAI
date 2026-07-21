#!/usr/bin/env python3
"""
Step 1 / Option 1 — Naive full-masked traversal: recall-collapse characterization.

No XQdrant Rust change required: the shipped ``focus.masked = true`` already masks
ALL HNSW layers, which *is* Option 1. This script sweeps the subspace ratio and
measures where recall collapses relative to full-vector ground truth.

Metric folder: experiments/<ts>__option1_naive_masked__recall_vs_ratio/

Run (live XQdrant):
    python run_option1_recall_collapse.py --mode http \\
        --xqdrant-url http://127.0.0.1:6333 --dimensions 768 --trials 5

Run (offline pipeline check):
    python run_option1_recall_collapse.py --mode simulated --dimensions 768 --queries 50 --trials 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import DEFAULT_EF_SEARCH, LatencyStats, TOP_K  # noqa: E402

STEP_ID = "option1_naive_masked"
METRIC = "recall_vs_ratio"
DEFAULT_RATIOS = [0.1, 0.25, 0.5, 0.75, 1.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["http", "simulated"], default="http")
    p.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    p.add_argument("--xqdrant-url", default=None)
    p.add_argument("--dimensions", type=int, nargs="+", default=[768])
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--k", type=int, default=TOP_K)
    p.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    p.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    p.add_argument("--no-provision", action="store_true")
    cr.add_dataset_args(p)
    cr.add_trials_arg(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    xqdrant_url = args.xqdrant_url or args.qdrant_url
    if args.trials < 1:
        print("ERROR: --trials must be >= 1", file=sys.stderr)
        return 2

    run = cr.init_research_run(
        STEP_ID, METRIC,
        mode=args.mode, dimensions=args.dimensions, queries=args.queries,
        qdrant_url=args.qdrant_url, xqdrant_url=xqdrant_url,
        requires_xqdrant_change="none (focus.masked already ships)",
        sweep={"ratios": args.ratios, "ef_search": args.ef_search, "k": args.k},
        extra={"dataset": args.dataset, "trials": args.trials},
    )
    print(f"[option1] experiment: {run.root}  trials={args.trials}")

    for dataset in cr.iter_datasets(args):
        dim = dataset.dimension
        norms = cr.full_norms_sq(dataset)

        client = None
        if args.mode == "http":
            client = cr.ResearchClient(
                dataset, args.qdrant_url, xqdrant_url, provision=not args.no_provision
            )
            warm_q, _ = cr.select_queries(dataset, min(200, args.queries), seed=cr.RANDOM_SEED)
            client.warmup(warm_q, len(warm_q))

        trial_rows: list[dict] = []
        for trial in range(args.trials):
            seed = cr.trial_seed(trial)
            queries, q_idx = cr.select_queries(dataset, args.queries, seed=seed)

            for ratio in args.ratios:
                dims = cr.subspace_indices(dim, ratio, seed=seed)
                lat = LatencyStats()
                recalls: list[float] = []
                unsupported = 0

                for local_i, q in enumerate(queries):
                    src = int(q_idx[local_i])
                    gt_ids = cr.full_top_k(dataset, src, q, args.k, norms)
                    if args.mode == "http":
                        body = cr.focus_body(q, args.k, args.ef_search, dims, masked=True)
                        out = client.query(body)
                        if out.status != "ok":
                            unsupported += 1
                            continue
                        lat.record(out.latency_ns)
                        recalls.append(cr.recall_at_k(out.result.ids, gt_ids, args.k))
                    else:
                        res, ns, _ = cr.simulated_masked(q, dataset, args.k, dims)
                        lat.record(ns)
                        recalls.append(cr.recall_at_k(res.ids, gt_ids, args.k))

                pct = lat.percentiles_ms()
                trial_rows.append({
                    "trial": trial,
                    "trial_seed": seed,
                    "dimension": dim,
                    "subspace_ratio": ratio,
                    "subspace_dims": len(dims),
                    "ef_search": args.ef_search,
                    "k": args.k,
                    "masked_recall_vs_full": float(np.mean(recalls)) if recalls else float("nan"),
                    "masked_p50_ms": pct["p50"],
                    "masked_p95_ms": pct["p95"],
                    "unsupported_queries": unsupported,
                    "simulated": int(args.mode == "simulated"),
                })
                tag = " [SIMULATED]" if args.mode == "simulated" else ""
                print(
                    f"  trial={trial} D={dim} ratio={ratio:>4}: "
                    f"recall={trial_rows[-1]['masked_recall_vs_full']:.3f} "
                    f"p50={pct['p50']:.3f}ms{tag}"
                    + (f"  ({unsupported} unsupported)" if unsupported else "")
                )

        csv_path = cr.bench_common.RESULTS_DIR / f"option1_recall_vs_ratio_d{dim}.csv"
        rows = cr.write_trial_and_summary_csv(
            csv_path,
            trial_rows,
            key_fields=["dimension", "subspace_ratio", "ef_search", "k"],
        )

        prefix = "[SIMULATED] " if args.mode == "simulated" else ""
        cr.line_plot(
            [r["subspace_ratio"] for r in rows],
            {"Masked recall@K (vs full GT)": [r["masked_recall_vs_full"] for r in rows]},
            yerr={
                "Masked recall@K (vs full GT)": [
                    r.get("masked_recall_vs_full_std", 0.0) for r in rows
                ]
            },
            xlabel="Subspace ratio (D_sub / D)",
            ylabel="Recall@K vs full-vector GT (mean ± std)",
            title=f"{prefix}M1 recall collapse (D={dim}, n={args.trials})",
            stem=f"option1_recall_vs_ratio_d{dim}",
            hline=0.8,
        )

    print(f"[option1] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
