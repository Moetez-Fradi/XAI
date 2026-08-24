#!/usr/bin/env python3
"""Fig 23 — MS MARCO / MiniLM case study: explain → restrict → re-search."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import DEFAULT_EF_SEARCH, MINILM_DIR, TOP_K  # noqa: E402

STEP_ID = "case_study"
METRIC = "marco_walkthrough"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["http", "simulated"], default="http")
    p.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    p.add_argument("--xqdrant-url", default=None)
    p.add_argument("--query-index", type=int, default=0)
    p.add_argument("--top-m", type=int, default=32)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--dimensions", type=int, nargs="+", default=[384])
    p.add_argument("--queries", type=int, default=8)
    p.add_argument("--no-provision", action="store_true")
    cr.add_dataset_args(p)
    return p.parse_args()


def _load_texts(minilm_dir: Path) -> list[str]:
    path = minilm_dir / "passages.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    args = parse_args()
    args.dataset = "minilm" if args.dataset == "synthetic" else args.dataset
    xq = args.xqdrant_url or args.qdrant_url
    run = cr.init_research_run(
        STEP_ID, METRIC, mode=args.mode, dimensions=args.dimensions,
        queries=args.queries, qdrant_url=args.qdrant_url, xqdrant_url=xq,
    )
    try:
        dataset = next(cr.iter_datasets(args))
    except FileNotFoundError:
        print("[case_study] MiniLM cache missing; using synthetic stand-in")
        args.dataset = "synthetic"
        args.dimensions = [384]
        dataset = next(cr.iter_datasets(args))

    q = dataset.queries[min(args.query_index, dataset.num_queries - 1)]
    texts = _load_texts(Path(getattr(args, "minilm_dir", str(MINILM_DIR))))
    client = None
    if args.mode == "http":
        client = cr.ResearchClient(
            dataset, args.qdrant_url, xq, provision=not args.no_provision
        )
        if not args.no_provision:
            client.wait_for_indexing(expected_points=dataset.num_vectors)

    if args.mode == "http":
        body = cr.focus_body(q, args.k, args.ef_search)
        body["with_dims_explained"] = {"top": args.top_m}
        full = client.query(body)
        if full.status != "ok" or not full.result:
            print("[case_study] full search failed:", full.detail)
            return 1
        full_ids = full.result.ids
        full_scores = full.result.scores
        terms = full.result.dims_explained[0] if full.result.dims_explained else {}
        focus_dims = np.array(sorted(terms, key=lambda d: -abs(terms[d]))[: args.top_m], dtype=np.int64)
        m2 = client.query(cr.focus_body(
            q, args.k, args.ef_search, focus_dims, masked=True, mask_from_layer=args.layer
        ))
        m2_ids = m2.result.ids if m2.status == "ok" and m2.result else []
        m2_scores = m2.result.scores if m2.status == "ok" and m2.result else []
    else:
        gt, sc = cr.brute_force_top_k(q, dataset.vectors, args.k, distance=dataset.distance)
        full_ids = [int(x) for x in gt]
        full_scores = [float(s) for s in sc]
        contrib = np.abs(q * dataset.vectors[full_ids[0]])
        focus_dims = np.argsort(-contrib)[: args.top_m]
        sub = cr.SubspaceGT(dataset.vectors, focus_dims, distance=dataset.distance)
        m2_ids_a, m2_sc = sub.top_k(q, args.k)
        m2_ids = [int(x) for x in m2_ids_a]
        m2_scores = [float(s) for s in m2_sc]
        terms = {int(d): float(contrib[d]) for d in focus_dims[:8]}

    overlap = set(full_ids) & set(m2_ids)
    rows = []
    for rank, pid in enumerate(full_ids):
        rows.append({
            "rank": rank, "id": int(pid), "full_score": float(full_scores[rank]),
            "in_m2_top": int(pid in overlap),
            "text": (texts[int(pid)][:80] if texts and int(pid) < len(texts) else ""),
        })
    cr.write_csv(cr.bench_common.RESULTS_DIR / "case_study_top5.csv",
                 list(rows[0].keys()), rows)

    plt = cr._mpl()
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    colors = ["#55a868" if r["in_m2_top"] else "#c44e52" for r in rows]
    ax.barh(range(len(rows)), [r["full_score"] for r in rows], color=colors)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"#{r['id']}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Full-search score")
    prefix = "[SIMULATED] " if args.mode == "simulated" else ""
    ax.set_title(f"{prefix}MS MARCO case study (green = also in M2 top-{args.k})")
    fig.tight_layout()
    cr.save_fig(fig, "fig23_case_study")
    print(f"[case_study] overlap={len(overlap)}/{args.k} -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
