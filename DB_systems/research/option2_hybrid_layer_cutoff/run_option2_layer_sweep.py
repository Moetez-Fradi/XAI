#!/usr/bin/env python3
"""
Step 2 / Option 2 — Hybrid: full-distance coarse layers, masked bottom layer(s).

Requires XQdrant Rust change: add ``focus.mask_from_layer``. Layer ``L`` uses masked
distance when ``L < mask_from_layer``, else full (``0`` = no masking / plain nearest;
``>= top_layer+1`` = all layers masked / Option 1). Omit with ``masked=true`` for legacy
full-mask behaviour.

This script does a 2D sweep ``mask_from_layer x D_sub/D`` and emits recall + latency
heatmaps with mean±std across ``--trials`` independent query-sample seeds.

Examples
--------
SIFT1M (D=128):
    python run_option2_layer_sweep.py --mode http \\
        --xqdrant-url http://127.0.0.1:6333 --dataset sift1m --no-provision \\
        --queries 500 --trials 5

High-D synthetic:
    python run_option2_layer_sweep.py --mode http \\
        --xqdrant-url http://127.0.0.1:6333 \\
        --dataset synthetic --dimensions 768 1536 \\
        --num-vectors 50000 --queries 500 --trials 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import DEFAULT_EF_SEARCH, LatencyStats, TOP_K  # noqa: E402

STEP_ID = "option2_hybrid_layer_cutoff"
METRIC = "recall_latency_layer_heatmap"
DEFAULT_RATIOS = [0.25, 0.5, 0.75]
DEFAULT_LAYERS = [0, 1, 2, 3]
DEFAULT_DIMENSIONS = [768, 1536]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["http", "simulated"], default="http")
    p.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    p.add_argument("--xqdrant-url", default=None)
    p.add_argument("--dimensions", type=int, nargs="+", default=DEFAULT_DIMENSIONS,
                   help="Synthetic dimensions to sweep (ignored for sift1m; default: 768 1536)")
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--k", type=int, default=TOP_K)
    p.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    p.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                   help="mask_from_layer values to sweep")
    p.add_argument("--no-provision", action="store_true",
                   help="Reuse an already-provisioned collection (skip create/upsert)")
    p.add_argument("--experiment-id", default=None,
                   help="Stable experiment folder name (default: auto timestamped)")
    cr.add_dataset_args(p)
    cr.add_trials_arg(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    xqdrant_url = args.xqdrant_url or args.qdrant_url
    if args.trials < 1:
        print("ERROR: --trials must be >= 1", file=sys.stderr)
        return 2

    datasets = list(cr.iter_datasets(args))
    actual_dims = [d.dimension for d in datasets]

    run = cr.init_research_run(
        STEP_ID, METRIC,
        mode=args.mode, dimensions=actual_dims, queries=args.queries,
        qdrant_url=args.qdrant_url, xqdrant_url=xqdrant_url,
        requires_xqdrant_change="focus.mask_from_layer (Option 2)",
        sweep={"ratios": args.ratios, "layers": args.layers,
               "ef_search": args.ef_search, "k": args.k},
        extra={
            "dataset": args.dataset,
            "num_vectors": args.num_vectors or (datasets[0].num_vectors if datasets else None),
            "requested_dimensions": list(args.dimensions),
            "trials": args.trials,
        },
        run_id=args.experiment_id,
    )
    print(f"[option2] experiment: {run.root}")
    print(f"[option2] dataset={args.dataset} dims={actual_dims} "
          f"N={datasets[0].num_vectors if datasets else '?'} Q={args.queries} "
          f"trials={args.trials}")

    for dataset in datasets:
        dim = dataset.dimension

        client = None
        if args.mode == "http":
            client = cr.ResearchClient(
                dataset, args.qdrant_url, xqdrant_url, provision=not args.no_provision
            )
            if not args.no_provision:
                print(f"  Waiting for index build (D={dim}, N={dataset.num_vectors})...")
                client.wait_for_indexing(expected_points=dataset.num_vectors)
            warm_q, _ = cr.select_queries(dataset, min(200, args.queries), seed=cr.RANDOM_SEED)
            client.warmup(warm_q, len(warm_q))

        trial_rows: list[dict] = []
        for trial in range(args.trials):
            seed = cr.trial_seed(trial)
            queries, _ = cr.select_queries(dataset, args.queries, seed=seed)

            # full-vector baseline p50 for speedup normalization (per trial)
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
                for ratio in args.ratios:
                    dims = cr.subspace_indices(dim, ratio, seed=seed)
                    subspace_gt = cr.SubspaceGT(dataset.vectors, dims, distance=dataset.distance)
                    lat = LatencyStats()
                    recalls: list[float] = []
                    unsupported = 0

                    for q in queries:
                        gt_ids, _ = subspace_gt.top_k(q, args.k)
                        if args.mode == "http":
                            body = cr.focus_body(
                                q, args.k, args.ef_search, dims,
                                masked=True, mask_from_layer=layer,
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
                            )
                            lat.record(ns)
                            recalls.append(cr.recall_at_k(res.ids, gt_ids, args.k))

                    p50 = lat.percentiles_ms()["p50"]
                    recall = float(np.mean(recalls)) if recalls else float("nan")
                    speedup = (full_p50 / p50) if p50 > 0 else float("nan")
                    trial_rows.append({
                        "trial": trial,
                        "trial_seed": seed,
                        "dimension": dim,
                        "mask_from_layer": layer,
                        "subspace_ratio": ratio,
                        "subspace_dims": len(dims),
                        "ef_search": args.ef_search,
                        "k": args.k,
                        "recall_at_k": recall,
                        "p50_ms": p50,
                        "full_p50_ms": full_p50,
                        "speedup_vs_full": speedup,
                        "unsupported_queries": unsupported,
                        "simulated": int(args.mode == "simulated"),
                    })
                    tag = " [SIM]" if args.mode == "simulated" else ""
                    print(
                        f"  trial={trial} D={dim} layer={layer} ratio={ratio}: "
                        f"recall={recall:.3f} speedup={speedup:.2f}x{tag}"
                        + (f"  ({unsupported} unsupported)" if unsupported else "")
                    )

        csv_path = cr.bench_common.RESULTS_DIR / f"option2_layer_sweep_d{dim}.csv"
        rows = cr.write_trial_and_summary_csv(
            csv_path,
            trial_rows,
            key_fields=["dimension", "mask_from_layer", "subspace_ratio", "ef_search", "k"],
        )

        layer_to_i = {L: i for i, L in enumerate(args.layers)}
        ratio_to_j = {R: j for j, R in enumerate(args.ratios)}
        recall_mat = np.full((len(args.layers), len(args.ratios)), np.nan)
        recall_std = np.full_like(recall_mat, np.nan)
        speed_mat = np.full_like(recall_mat, np.nan)
        speed_std = np.full_like(recall_mat, np.nan)
        for r in rows:
            i = layer_to_i[int(r["mask_from_layer"])]
            j = ratio_to_j[float(r["subspace_ratio"])]
            recall_mat[i, j] = float(r["recall_at_k"])
            recall_std[i, j] = float(r.get("recall_at_k_std", 0.0) or 0.0)
            speed_mat[i, j] = float(r["speedup_vs_full"])
            speed_std[i, j] = float(r.get("speedup_vs_full_std", 0.0) or 0.0)

        prefix = "[SIMULATED] " if args.mode == "simulated" else ""
        cr.heatmap(
            recall_mat, row_labels=args.layers, col_labels=args.ratios,
            xlabel="Subspace ratio (D_sub / D)", ylabel="mask_from_layer",
            title=f"{prefix}Option 2 recall@K (D={dim}, n={args.trials})",
            stem=f"option2_recall_heatmap_d{dim}", cbar_label="Recall@K (mean)",
            fmt="{:.2f}", std_matrix=recall_std,
        )
        cr.heatmap(
            speed_mat, row_labels=args.layers, col_labels=args.ratios,
            xlabel="Subspace ratio (D_sub / D)", ylabel="mask_from_layer",
            title=f"{prefix}Option 2 speedup vs full (D={dim}, n={args.trials})",
            stem=f"option2_speedup_heatmap_d{dim}", cbar_label="Speedup (x, mean)",
            fmt="{:.2f}", std_matrix=speed_std,
        )

    print(f"[option2] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
