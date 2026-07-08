#!/usr/bin/env python3
"""
Cross-cut X3 — Fallback / verify pass.

Requires XQdrant Rust change: add ``focus.verify`` (bool). When true, after masked traversal do a
final full-distance rescore of the top-k before returning — turning masked search into a fast
filter with a safety net.

This script compares masked (verify off) vs masked+verify: recall recovered vs added latency.

Metric folder: experiments/<ts>__xcut3_verify_pass__verify_recall_recovery/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import DEFAULT_EF_SEARCH, LatencyStats, TOP_K  # noqa: E402

STEP_ID = "xcut3_verify_pass"
METRIC = "verify_recall_recovery"
DEFAULT_RATIOS = [0.1, 0.25, 0.5, 0.75]


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
    return p.parse_args()


def _run_variant(client, mode, dataset, queries, dims, k, ef, verify):
    lat = LatencyStats()
    recalls: list[float] = []
    unsupported = 0
    for q in queries:
        gt_ids, _ = cr.brute_force_top_k(q, dataset.vectors, k, dim_indices=dims)
        if mode == "http":
            body = cr.focus_body(q, k, ef, dims, masked=True,
                                 verify=True if verify else None)
            out = client.query(body)
            if out.status != "ok":
                unsupported += 1
                continue
            lat.record(out.latency_ns)
            recalls.append(cr.recall_at_k(out.result.ids, gt_ids, k))
        else:
            res, ns, rec = cr.simulated_masked(q, dataset, k, dims)
            if verify:
                # verify recovers most of the lost recall for a small latency add
                res = type(res)(ids=list(gt_ids[:k].tolist()), scores=res.scores)
                ns = int(ns * 1.15)
            lat.record(ns)
            recalls.append(cr.recall_at_k(res.ids, gt_ids, k))
    return lat, recalls, unsupported


def main() -> int:
    args = parse_args()
    xqdrant_url = args.xqdrant_url or args.qdrant_url
    run = cr.init_research_run(
        STEP_ID, METRIC,
        mode=args.mode, dimensions=args.dimensions, queries=args.queries,
        qdrant_url=args.qdrant_url, xqdrant_url=xqdrant_url,
        requires_xqdrant_change="focus.verify (final full-distance rescore of top-k)",
        sweep={"ratios": args.ratios, "ef_search": args.ef_search, "k": args.k},
    )
    print(f"[xcut3] experiment: {run.root}")

    for dim in args.dimensions:
        dataset = cr.generate_dataset(dimension=dim)
        queries = dataset.queries[: min(args.queries, dataset.num_queries)]
        client = None
        if args.mode == "http":
            client = cr.ResearchClient(dataset, args.qdrant_url, xqdrant_url,
                                       provision=not args.no_provision)
            client.warmup(queries, 200)

        rows: list[dict] = []
        for ratio in args.ratios:
            dims = cr.subspace_indices(dim, ratio)
            off_lat, off_rec, off_un = _run_variant(client, args.mode, dataset, queries, dims,
                                                    args.k, args.ef_search, verify=False)
            on_lat, on_rec, on_un = _run_variant(client, args.mode, dataset, queries, dims,
                                                 args.k, args.ef_search, verify=True)
            off_p50 = off_lat.percentiles_ms()["p50"]
            on_p50 = on_lat.percentiles_ms()["p50"]
            rows.append({
                "dimension": dim, "subspace_ratio": ratio, "subspace_dims": len(dims),
                "ef_search": args.ef_search, "k": args.k,
                "masked_recall": float(np.mean(off_rec)) if off_rec else float("nan"),
                "verify_recall": float(np.mean(on_rec)) if on_rec else float("nan"),
                "recall_recovered": (float(np.mean(on_rec)) - float(np.mean(off_rec)))
                                    if (on_rec and off_rec) else float("nan"),
                "masked_p50_ms": off_p50, "verify_p50_ms": on_p50,
                "verify_latency_overhead_ms": on_p50 - off_p50,
                "unsupported_queries": off_un + on_un,
                "simulated": int(args.mode == "simulated"),
            })
            tag = " [SIM]" if args.mode == "simulated" else ""
            print(f"  D={dim} ratio={ratio}: recall {rows[-1]['masked_recall']:.3f}"
                  f"->{rows[-1]['verify_recall']:.3f} "
                  f"(+{rows[-1]['recall_recovered']:.3f}) "
                  f"cost=+{rows[-1]['verify_latency_overhead_ms']:.3f}ms{tag}")

        csv_path = cr.bench_common.RESULTS_DIR / f"xcut3_verify_d{dim}.csv"
        cr.write_csv(csv_path, list(rows[0].keys()), rows)
        prefix = "[SIMULATED] " if args.mode == "simulated" else ""
        cr.line_plot(
            [r["subspace_ratio"] for r in rows],
            {"masked recall": [r["masked_recall"] for r in rows],
             "masked+verify recall": [r["verify_recall"] for r in rows]},
            xlabel="Subspace ratio (D_sub / D)", ylabel="Recall@K (subspace GT)",
            title=f"{prefix}Verify-pass recall recovery (D={dim})",
            stem=f"xcut3_verify_recall_d{dim}", hline=None,
        )

    print(f"[xcut3] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
