#!/usr/bin/env python3
"""Fig 8 + Table 2 — MiniLM / GIST M1 recall vs subspace ratio (HTTP).

Sweeps D_sub/D on MiniLM (D=384) and/or GIST (D=960) with naive all-layer masking.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import DEFAULT_EF_SEARCH, LatencyStats, TOP_K  # noqa: E402

STEP_ID = "cross_corpus"
METRIC = "m1_recall_vs_ratio"
DEFAULT_RATIOS = [0.1, 0.25, 0.5, 0.75, 1.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["http", "simulated"], default="http")
    p.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    p.add_argument("--xqdrant-url", default=None)
    p.add_argument("--corpora", nargs="+", default=["minilm", "gist1m"],
                   choices=["minilm", "gist1m", "sift1m", "synthetic"])
    p.add_argument("--dimensions", type=int, nargs="+", default=[384])
    p.add_argument("--queries", type=int, default=40)
    p.add_argument("--k", type=int, default=TOP_K)
    p.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    p.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    p.add_argument("--no-provision", action="store_true")
    cr.add_dataset_args(p)
    cr.add_trials_arg(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    xq = args.xqdrant_url or args.qdrant_url
    run = cr.init_research_run(
        STEP_ID, METRIC, mode=args.mode, dimensions=args.dimensions,
        queries=args.queries, qdrant_url=args.qdrant_url, xqdrant_url=xq,
        sweep={"ratios": args.ratios, "corpora": args.corpora},
        extra={"trials": args.trials},
    )
    all_rows = []
    for corpus in args.corpora:
        args.dataset = corpus
        for dataset in cr.iter_datasets(args):
            dim = dataset.dimension
            norms = cr.full_norms_sq(dataset)
            client = None
            if args.mode == "http":
                client = cr.ResearchClient(
                    dataset, args.qdrant_url, xq, provision=not args.no_provision
                )
                if not args.no_provision:
                    client.wait_for_indexing(expected_points=dataset.num_vectors)
            trial_rows = []
            for trial in range(args.trials):
                seed = cr.trial_seed(trial)
                queries, q_idx = cr.select_queries(dataset, args.queries, seed=seed)
                for ratio in args.ratios:
                    dims = cr.subspace_indices(dim, ratio, seed=seed)
                    lat = LatencyStats()
                    recalls = []
                    unsupported = 0
                    for local_i, q in enumerate(queries):
                        src = int(q_idx[local_i])
                        gt = cr.full_top_k(dataset, src, q, args.k, norms)
                        if args.mode == "http":
                            out = client.query(
                                cr.focus_body(q, args.k, args.ef_search, dims, masked=True)
                            )
                            if out.status != "ok":
                                unsupported += 1
                                continue
                            lat.record(out.latency_ns)
                            recalls.append(cr.recall_at_k(out.result.ids, gt, args.k))
                        else:
                            res, ns, _ = cr.simulated_masked(q, dataset, args.k, dims)
                            lat.record(ns)
                            recalls.append(cr.recall_at_k(res.ids, gt, args.k))
                    trial_rows.append({
                        "trial": trial, "corpus": corpus, "dimension": dim,
                        "subspace_ratio": ratio,
                        "masked_recall_vs_full": float(np.mean(recalls)) if recalls else float("nan"),
                        "p50_ms": lat.percentiles_ms()["p50"],
                        "unsupported_queries": unsupported,
                        "simulated": int(args.mode == "simulated"),
                    })
                    print(f"  {corpus} trial={trial} r={ratio}: "
                          f"recall={trial_rows[-1]['masked_recall_vs_full']:.3f}")
            csv = cr.bench_common.RESULTS_DIR / f"m1_{corpus}_d{dim}.csv"
            rows = cr.write_trial_and_summary_csv(
                csv, trial_rows, key_fields=["corpus", "dimension", "subspace_ratio"]
            )
            all_rows.extend(rows)
            prefix = "[SIMULATED] " if args.mode == "simulated" else ""
            cr.line_plot(
                [r["subspace_ratio"] for r in rows],
                {f"{corpus} Recall@10": [r["masked_recall_vs_full"] for r in rows]},
                yerr={f"{corpus} Recall@10": [r.get("masked_recall_vs_full_std", 0.0) for r in rows]},
                xlabel=r"$D_{\mathrm{sub}}/D$", ylabel="Recall@10 vs full GT",
                title=f"{prefix}M1 {corpus} (D={dim}, T={args.trials})",
                stem=f"fig08_{corpus}_m1",
                hline=1.0,
            )
    print(f"[cross_corpus] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
