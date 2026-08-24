#!/usr/bin/env python3
"""
Step 4 / Option 4 — Weighted blend distance: d = alpha*d_full + (1-alpha)*d_focus.

Requires XQdrant Rust change: add ``focus.alpha`` (f32). Note this alone does NOT save
full-distance computation (both distances are computed); it only biases routing, so pair it with
Option 2's ``mask_from_layer`` for real latency gains. Treat as a tuning knob for Option 1/2.

This script sweeps ``alpha x mask_from_layer`` and reports recall + speedup with mean±std
across ``--trials`` independent query-sample seeds.

Metric folder: experiments/<ts>__option4_weighted_blend__recall_speed_alpha/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import DEFAULT_EF_SEARCH, LatencyStats, TOP_K  # noqa: E402

STEP_ID = "option4_weighted_blend"
METRIC = "recall_speed_alpha"
DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
# Pair with Option 2 cutoffs: L=0 (no mask) plus hybrid L>=1 where alpha matters.
DEFAULT_LAYERS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["http", "simulated"], default="http")
    p.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    p.add_argument("--xqdrant-url", default=None)
    p.add_argument("--dimensions", type=int, nargs="+", default=[768])
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--k", type=int, default=TOP_K)
    p.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    p.add_argument("--ratio", type=float, default=0.25, help="fixed subspace ratio for the sweep")
    p.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
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
        requires_xqdrant_change="focus.alpha (Option 4); pair with focus.mask_from_layer",
        sweep={"ratio": args.ratio, "alphas": args.alphas, "layers": args.layers,
               "ef_search": args.ef_search, "k": args.k},
        extra={"dataset": args.dataset, "trials": args.trials},
    )
    print(f"[option4] experiment: {run.root}  trials={args.trials}")

    for dataset in cr.iter_datasets(args):
        dim = dataset.dimension

        client = None
        if args.mode == "http":
            client = cr.ResearchClient(
                dataset, args.qdrant_url, xqdrant_url, provision=not args.no_provision,
            )
            warm_q, _ = cr.select_queries(dataset, min(200, args.queries), seed=cr.RANDOM_SEED)
            client.warmup(warm_q, len(warm_q))

        trial_rows: list[dict] = []
        for trial in range(args.trials):
            seed = cr.trial_seed(trial)
            queries, _ = cr.select_queries(dataset, args.queries, seed=seed)
            dims = cr.subspace_indices(dim, args.ratio, seed=seed)
            subspace_gt = cr.SubspaceGT(dataset.vectors, dims, distance=dataset.distance)

            base_lat = LatencyStats()
            for q in queries:
                if args.mode == "http":
                    out = client.query(cr.focus_body(q, args.k, args.ef_search))
                    if out.status == "ok":
                        base_lat.record(out.latency_ns)
                else:
                    base_lat.record(1_000_000)
            full_p50 = base_lat.percentiles_ms()["p50"] or 1.0

            for layer in args.layers:
                for alpha in args.alphas:
                    lat = LatencyStats()
                    recalls: list[float] = []
                    unsupported = 0
                    for q in queries:
                        gt_ids, _ = subspace_gt.top_k(q, args.k)
                        if args.mode == "http":
                            body = cr.focus_body(
                                q, args.k, args.ef_search, dims,
                                masked=True, mask_from_layer=layer, alpha=alpha,
                            )
                            out = client.query(body)
                            if out.status != "ok":
                                unsupported += 1
                                continue
                            lat.record(out.latency_ns)
                            recalls.append(cr.recall_at_k(out.result.ids, gt_ids, args.k))
                        else:
                            res, ns, _ = cr.simulated_masked(
                                q, dataset, args.k, dims,
                                mask_from_layer=layer,
                                max_layer=max(args.layers) or 1,
                                alpha=alpha,
                            )
                            lat.record(ns)
                            recalls.append(cr.recall_at_k(res.ids, gt_ids, args.k))

                    p50 = lat.percentiles_ms()["p50"]
                    recall = float(np.mean(recalls)) if recalls else float("nan")
                    trial_rows.append({
                        "trial": trial,
                        "trial_seed": seed,
                        "dimension": dim,
                        "subspace_ratio": args.ratio,
                        "mask_from_layer": layer,
                        "alpha": alpha,
                        "ef_search": args.ef_search,
                        "k": args.k,
                        "recall_at_k": recall,
                        "p50_ms": p50,
                        "full_p50_ms": full_p50,
                        "speedup_vs_full": (full_p50 / p50) if p50 > 0 else float("nan"),
                        "unsupported_queries": unsupported,
                        "simulated": int(args.mode == "simulated"),
                    })
                    tag = " [SIM]" if args.mode == "simulated" else ""
                    print(
                        f"  trial={trial} D={dim} layer={layer} alpha={alpha}: "
                        f"recall={recall:.3f}{tag}"
                        + (f"  ({unsupported} unsupported)" if unsupported else "")
                    )

        csv_path = cr.bench_common.RESULTS_DIR / f"option4_alpha_sweep_d{dim}.csv"
        rows = cr.write_trial_and_summary_csv(
            csv_path,
            trial_rows,
            key_fields=["dimension", "subspace_ratio", "mask_from_layer", "alpha", "ef_search", "k"],
        )

        prefix = "[SIMULATED] " if args.mode == "simulated" else ""
        for layer in args.layers:
            lr = sorted(
                [r for r in rows if int(r["mask_from_layer"]) == layer],
                key=lambda r: float(r["alpha"]),
            )
            label = f"recall (mask_from_layer={layer})"
            cr.line_plot(
                [float(r["alpha"]) for r in lr],
                {label: [float(r["recall_at_k"]) for r in lr]},
                yerr={
                    label: [float(r.get("recall_at_k_std", 0.0) or 0.0) for r in lr],
                },
                xlabel="alpha (0 = focus only, 1 = full distance)",
                ylabel="Recall@K (subspace GT, mean ± std)",
                title=f"{prefix}M3 blend recall (D={dim}, ratio={args.ratio}, n={args.trials})",
                stem=f"option4_alpha_recall_d{dim}_layer{layer}",
                hline=None,
            )

    print(f"[option4] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
