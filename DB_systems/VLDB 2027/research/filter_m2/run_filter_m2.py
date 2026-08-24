#!/usr/bin/env python3
"""Fig 20 — payload filter × M2: recall/latency vs selectivity {0.1, 0.5, 1.0}."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import DEFAULT_EF_SEARCH, LatencyStats, TOP_K  # noqa: E402

STEP_ID = "filter_m2"
METRIC = "selectivity_recall_latency"
DEFAULT_SELS = [0.1, 0.5, 1.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["http", "simulated"], default="http")
    p.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    p.add_argument("--xqdrant-url", default=None)
    p.add_argument("--dimensions", type=int, nargs="+", default=[128])
    p.add_argument("--queries", type=int, default=40)
    p.add_argument("--k", type=int, default=TOP_K)
    p.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--selectivities", type=float, nargs="+", default=DEFAULT_SELS)
    p.add_argument("--no-provision", action="store_true")
    cr.add_dataset_args(p)
    cr.add_trials_arg(p)
    return p.parse_args()


def _filtered_gt(dataset, query, k, norms, filt_ids: set[int]):
    if not filt_ids:
        return np.array([], dtype=np.int64)
    mask = np.array([int(i) in filt_ids for i in dataset.ids], dtype=bool)
    sub_vecs = dataset.vectors[mask]
    sub_ids = dataset.ids[mask]
    ids, _ = cr.brute_force_top_k(
        query, sub_vecs, min(k, len(sub_ids)),
        distance=dataset.distance, vectors_norm_sq=None if norms is None else norms[mask],
    )
    return np.asarray(sub_ids[ids], dtype=np.int64)


def main() -> int:
    args = parse_args()
    xq = args.xqdrant_url or args.qdrant_url
    run = cr.init_research_run(
        STEP_ID, METRIC, mode=args.mode, dimensions=args.dimensions,
        queries=args.queries, qdrant_url=args.qdrant_url, xqdrant_url=xq,
        sweep={"selectivities": args.selectivities, "ratio": args.ratio},
        extra={"trials": args.trials},
    )
    trial_rows = []
    for dataset in cr.iter_datasets(args):
        dim = dataset.dimension
        norms = cr.full_norms_sq(dataset)
        client = None
        if args.mode == "http":
            client = cr.ResearchClient(
                dataset, args.qdrant_url, xq,
                provision=not args.no_provision,
                with_payload_buckets=True,
            )
            if not args.no_provision:
                client.wait_for_indexing(expected_points=dataset.num_vectors)
        for trial in range(args.trials):
            seed = cr.trial_seed(trial)
            queries, q_idx = cr.select_queries(dataset, args.queries, seed=seed)
            dims = cr.subspace_indices(dim, args.ratio, seed=seed)
            for sel in args.selectivities:
                filt = cr.payload_filter(sel)
                keep = max(1, int(round(sel * 10)))
                allowed = set(int(i) for i in dataset.ids if int(i) % 10 < keep)
                lat = LatencyStats()
                recalls = []
                for local_i, q in enumerate(queries):
                    gt = _filtered_gt(dataset, q, args.k, norms, allowed)
                    if args.mode == "http":
                        body = cr.attach_filter(
                            cr.focus_body(
                                q, args.k, args.ef_search, dims,
                                masked=True, mask_from_layer=args.layer,
                            ),
                            filt,
                        )
                        out = client.query(body)
                        if out.status != "ok":
                            continue
                        lat.record(out.latency_ns)
                        recalls.append(cr.recall_at_k(out.result.ids, gt, args.k))
                    else:
                        rec = float(np.clip(0.91 - 0.15 * (1.0 - sel), 0.4, 1.0))
                        recalls.append(rec)
                        lat.record(int((0.8 + 0.4 * (1.0 - sel)) * 1e6))
                trial_rows.append({
                    "trial": trial, "dimension": dim, "selectivity": sel,
                    "ratio": args.ratio,
                    "recall_at_k": float(np.mean(recalls)) if recalls else float("nan"),
                    "p50_ms": lat.percentiles_ms()["p50"],
                    "simulated": int(args.mode == "simulated"),
                })
                print(f"  sel={sel} trial={trial}: recall={trial_rows[-1]['recall_at_k']:.3f}")

    rows = cr.write_trial_and_summary_csv(
        cr.bench_common.RESULTS_DIR / "filter_m2.csv",
        trial_rows, key_fields=["dimension", "selectivity", "ratio"],
    )
    plt = cr._mpl()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    xs = [float(r["selectivity"]) for r in rows]
    axes[0].errorbar(
        xs, [float(r["recall_at_k"]) for r in rows],
        yerr=[float(r.get("recall_at_k_std", 0.0) or 0.0) for r in rows],
        marker="o",
    )
    axes[1].errorbar(
        xs, [float(r["p50_ms"]) for r in rows],
        yerr=[float(r.get("p50_ms_std", 0.0) or 0.0) for r in rows],
        marker="o", color="#c44e52",
    )
    axes[0].set_xlabel("Filter selectivity")
    axes[0].set_ylabel("Recall@10")
    axes[1].set_xlabel("Filter selectivity")
    axes[1].set_ylabel("p50 ms")
    prefix = "[SIMULATED] " if args.mode == "simulated" else ""
    fig.suptitle(f"{prefix}Filter × M2", y=1.02)
    fig.tight_layout()
    cr.save_fig(fig, "fig20_filter_m2")
    print(f"[filter_m2] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
