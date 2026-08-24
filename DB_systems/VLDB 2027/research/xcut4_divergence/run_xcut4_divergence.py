#!/usr/bin/env python3
"""
Cross-cut X4 — Candidate-list divergence.

Ideal metric: how often the masked-traversal *visited-node* set diverges from the full-traversal
visited-node set. That needs XQdrant instrumentation (a logged/returned divergence count) — plumb
it during the Option 1/2 runs (no separate experiment). Pass that CSV via ``--divergence-csv`` to
plot it.

Runnable proxy now (no XQdrant change): the divergence of the *returned top-k* sets between masked
and full search across subspace ratios. This already shows where masking starts steering results
away from the full-space answer.

Metric folder: experiments/<ts>__xcut4_divergence__traversal_divergence/
"""

from __future__ import annotations

import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import DEFAULT_EF_SEARCH, TOP_K  # noqa: E402

STEP_ID = "xcut4_divergence"
METRIC = "traversal_divergence"
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
    p.add_argument("--layer", type=int, default=1, help="M2 mask_from_layer for Jaccard overlay")
    p.add_argument("--divergence-csv", type=Path, default=None,
                   help="Optional XQdrant-emitted visited-node divergence CSV "
                        "(schema: subspace_ratio,visited_divergence)")
    p.add_argument("--no-provision", action="store_true")
    cr.add_dataset_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    xqdrant_url = args.xqdrant_url or args.qdrant_url
    run = cr.init_research_run(
        STEP_ID, METRIC,
        mode=args.mode, dimensions=args.dimensions, queries=args.queries,
        qdrant_url=args.qdrant_url, xqdrant_url=xqdrant_url,
        requires_xqdrant_change="visited-node divergence logging (proxy: returned-set divergence needs none)",
        sweep={"ratios": args.ratios, "ef_search": args.ef_search, "k": args.k},
        extra={"dataset": args.dataset},
    )
    print(f"[xcut4] experiment: {run.root}")

    instrumentation = {}
    if args.divergence_csv and args.divergence_csv.exists():
        for r in _csv.DictReader(args.divergence_csv.open(encoding="utf-8")):
            instrumentation[float(r["subspace_ratio"])] = float(r["visited_divergence"])

    for dataset in cr.iter_datasets(args):
        dim = dataset.dimension
        queries = dataset.queries[: min(args.queries, dataset.num_queries)]
        norms = cr.full_norms_sq(dataset)
        client = None
        if args.mode == "http":
            client = cr.ResearchClient(dataset, args.qdrant_url, xqdrant_url,
                                       provision=not args.no_provision)
            client.warmup(queries, 200)

        rows: list[dict] = []
        for ratio in args.ratios:
            dims = cr.subspace_indices(dim, ratio)
            m1_jac, m2_jac = [], []
            unsupported = 0

            def _jaccard(a: set[int], b: set[int]) -> float:
                if not a and not b:
                    return 1.0
                return len(a & b) / max(1, len(a | b))

            for i, q in enumerate(queries):
                if args.mode == "http":
                    full_out = client.query(cr.focus_body(q, args.k, args.ef_search))
                    m1_out = client.query(
                        cr.focus_body(q, args.k, args.ef_search, dims, masked=True))
                    m2_out = client.query(
                        cr.focus_body(q, args.k, args.ef_search, dims, masked=True,
                                      mask_from_layer=args.layer))
                    if full_out.status != "ok" or m1_out.status != "ok":
                        unsupported += 1
                        continue
                    full_set = set(full_out.result.ids[: args.k])
                    m1_jac.append(_jaccard(full_set, set(m1_out.result.ids[: args.k])))
                    if m2_out.status == "ok":
                        m2_jac.append(_jaccard(full_set, set(m2_out.result.ids[: args.k])))
                else:
                    full_ids = cr.full_top_k(dataset, i, q, args.k, norms)
                    res1, _, _ = cr.simulated_masked(q, dataset, args.k, dims)
                    res2, _, _ = cr.simulated_masked(
                        q, dataset, args.k, dims, mask_from_layer=args.layer, max_layer=4
                    )
                    full_set = set(int(x) for x in full_ids)
                    m1_jac.append(_jaccard(full_set, set(res1.ids[: args.k])))
                    m2_jac.append(_jaccard(full_set, set(res2.ids[: args.k])))

            row = {
                "dimension": dim, "subspace_ratio": ratio, "subspace_dims": len(dims),
                "ef_search": args.ef_search, "k": args.k,
                "m1_jaccard": float(np.mean(m1_jac)) if m1_jac else float("nan"),
                "m2_jaccard": float(np.mean(m2_jac)) if m2_jac else float("nan"),
                "returned_set_divergence": (
                    1.0 - float(np.mean(m1_jac)) if m1_jac else float("nan")
                ),
                "unsupported_queries": unsupported,
                "simulated": int(args.mode == "simulated"),
            }
            if ratio in instrumentation:
                row["visited_node_divergence"] = instrumentation[ratio]
            rows.append(row)
            tag = " [SIM]" if args.mode == "simulated" else ""
            print(f"  D={dim} ratio={ratio}: M1 Jaccard={row['m1_jaccard']:.3f} "
                  f"M2 Jaccard={row['m2_jaccard']:.3f}{tag}")

        csv_path = cr.bench_common.RESULTS_DIR / f"xcut4_divergence_d{dim}.csv"
        cr.write_csv(csv_path, list(rows[0].keys()), rows)

        series = {
            "M1 Jaccard": [r["m1_jaccard"] for r in rows],
            "M2 Jaccard": [r["m2_jaccard"] for r in rows],
        }
        prefix = "[SIMULATED] " if args.mode == "simulated" else ""
        cr.line_plot(
            [r["subspace_ratio"] for r in rows], series,
            xlabel="Subspace ratio (D_sub / D)", ylabel="Top-k Jaccard vs full search",
            title=f"{prefix}X4 result-set Jaccard (D={dim})",
            stem="fig17_x4_jaccard",
        )

    print(f"[xcut4] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
