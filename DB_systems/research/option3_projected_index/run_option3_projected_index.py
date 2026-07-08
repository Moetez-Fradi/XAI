#!/usr/bin/env python3
"""
Step 3 / Option 3 — Auxiliary projected index for common focus sets.

Full XQdrant Rust change (index registration + query routing) is the most invasive of the four.
BUT the key quantity this step proves — the **recall ceiling** of a real HNSW built *in* the
subspace, plus its **memory** and **build-time** cost — can be measured today by building a
stock Qdrant collection over the projected (focus-dim-only) vectors. That collection IS
"a from-scratch full HNSW build in that subspace" from steps_vdbt.md.

So this script needs no XQdrant change to produce the ceiling numbers; the Rust work is only to
make it a routed, first-class feature.

Metric folder: experiments/<ts>__option3_projected_index__recall_memory_buildtime/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import DEFAULT_EF_SEARCH, LatencyStats, TOP_K  # noqa: E402

STEP_ID = "option3_projected_index"
METRIC = "recall_memory_buildtime"
DEFAULT_RATIOS = [0.25, 0.5, 0.75]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["http", "simulated"], default="http")
    p.add_argument("--xqdrant-url", default="http://127.0.0.1:6333")
    p.add_argument("--dimensions", type=int, nargs="+", default=[768])
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--k", type=int, default=TOP_K)
    p.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    p.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    return p.parse_args()


def _req(session, url, method, path, body=None):
    r = session.request(method, f"{url}{path}", json=body, timeout=120)
    if not r.ok:
        raise requests.HTTPError(f"{r.status_code} {r.reason} for {method} {path}: {r.text[:300]}")
    return r.json()


def _build_projected_collection(session, url, name, vectors, ids) -> float:
    """Create a stock collection over projected vectors; return build wall-time (s)."""
    dsub = int(vectors.shape[1])
    try:
        _req(session, url, "DELETE", f"/collections/{name}")
    except requests.HTTPError:
        pass
    _req(session, url, "PUT", f"/collections/{name}",
         {"vectors": {"size": dsub, "distance": "Dot"},
          "hnsw_config": {"m": 16, "ef_construct": 200}})
    start = time.perf_counter()
    batch = 256
    for s in range(0, len(ids), batch):
        e = min(s + batch, len(ids))
        pts = [{"id": int(ids[i]), "vector": vectors[i].tolist()} for i in range(s, e)]
        _req(session, url, "PUT", f"/collections/{name}/points?wait=true", {"points": pts})
    return time.perf_counter() - start


def main() -> int:
    args = parse_args()
    run = cr.init_research_run(
        STEP_ID, METRIC,
        mode=args.mode, dimensions=args.dimensions, queries=args.queries,
        xqdrant_url=args.xqdrant_url,
        requires_xqdrant_change="projected-index build path + query routing (ceiling measurable now via stock collection)",
        sweep={"ratios": args.ratios, "ef_search": args.ef_search, "k": args.k},
    )
    print(f"[option3] experiment: {run.root}")

    for dim in args.dimensions:
        dataset = cr.generate_dataset(dimension=dim)
        queries = dataset.queries[: min(args.queries, dataset.num_queries)]
        session = requests.Session() if args.mode == "http" else None
        rows: list[dict] = []

        for ratio in args.ratios:
            dims = cr.subspace_indices(dim, ratio)
            dsub = len(dims)
            mem_bytes = dataset.num_vectors * dsub * 4  # f32 vector-store proxy

            if args.mode == "http":
                name = f"proj_{dataset.name}_r{int(ratio * 100)}"
                proj_vecs = np.ascontiguousarray(dataset.vectors[:, dims])
                build_s = _build_projected_collection(session, args.xqdrant_url, name,
                                                       proj_vecs, dataset.ids)
                lat = LatencyStats()
                recalls: list[float] = []
                for q in queries:
                    gt_ids, _ = cr.brute_force_top_k(q, dataset.vectors, args.k, dim_indices=dims)
                    sub_q = np.ascontiguousarray(q[dims])
                    start = time.perf_counter_ns()
                    payload = _req(session, args.xqdrant_url, "POST",
                                   f"/collections/{name}/points/query",
                                   {"query": {"nearest": sub_q.tolist()}, "limit": args.k,
                                    "params": {"hnsw_ef": args.ef_search}})
                    lat.record(time.perf_counter_ns() - start)
                    got = [int(p["id"]) for p in payload.get("result", {}).get("points", [])]
                    recalls.append(cr.recall_at_k(got, gt_ids, args.k))
                try:
                    _req(session, args.xqdrant_url, "DELETE", f"/collections/{name}")
                except requests.HTTPError:
                    pass
                recall = float(np.mean(recalls))
                p50 = lat.percentiles_ms()["p50"]
            else:
                build_s = dataset.num_vectors * dsub * 1e-7  # simulated build cost
                recall = float(np.clip(0.99 - 0.02 * (1 - ratio), 0, 1))  # near-ceiling
                p50 = max(0.2, dsub / dim) * 1.0

            rows.append({
                "dimension": dim, "subspace_ratio": ratio, "subspace_dims": dsub,
                "ef_search": args.ef_search, "k": args.k,
                "projected_recall_at_k": recall, "projected_p50_ms": p50,
                "build_time_s": build_s, "memory_proxy_bytes": mem_bytes,
                "memory_proxy_mb": mem_bytes / 1e6,
                "simulated": int(args.mode == "simulated"),
            })
            tag = " [SIM]" if args.mode == "simulated" else ""
            print(f"  D={dim} ratio={ratio}: recall={recall:.3f} build={build_s:.2f}s "
                  f"mem~{mem_bytes / 1e6:.1f}MB{tag}")

        csv_path = cr.bench_common.RESULTS_DIR / f"option3_projected_index_d{dim}.csv"
        cr.write_csv(csv_path, list(rows[0].keys()), rows)

        prefix = "[SIMULATED] " if args.mode == "simulated" else ""
        cr.line_plot(
            [r["subspace_ratio"] for r in rows],
            {"Projected-index recall@K (ceiling)": [r["projected_recall_at_k"] for r in rows]},
            xlabel="Subspace ratio (D_sub / D)", ylabel="Recall@K (subspace GT)",
            title=f"{prefix}Option 3 recall ceiling (D={dim})",
            stem=f"option3_recall_ceiling_d{dim}", hline=1.0,
        )
        cr.line_plot(
            [r["subspace_ratio"] for r in rows],
            {"Build time (s)": [r["build_time_s"] for r in rows],
             "Memory proxy (MB)": [r["memory_proxy_mb"] for r in rows]},
            xlabel="Subspace ratio (D_sub / D)", ylabel="Cost",
            title=f"{prefix}Option 3 build/memory cost (D={dim})",
            stem=f"option3_cost_d{dim}",
        )

    print(f"[option3] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
