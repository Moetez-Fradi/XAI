#!/usr/bin/env python3
"""Fig 13 — post-hoc rescore vs M1 vs M2 under random / contiguous / top-energy focuses."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import DEFAULT_EF_SEARCH, LatencyStats, TOP_K  # noqa: E402

STEP_ID = "focus_baselines"
METRIC = "posthoc_m1_m2"
CONSTRUCTIONS = ("random", "contiguous", "top_energy")
METHODS = ("posthoc", "m1", "m2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["http", "simulated"], default="http")
    p.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    p.add_argument("--xqdrant-url", default=None)
    p.add_argument("--dimensions", type=int, nargs="+", default=[128])
    p.add_argument("--queries", type=int, default=40)
    p.add_argument("--k", type=int, default=TOP_K)
    p.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    p.add_argument("--ratio", type=float, default=0.25)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--no-provision", action="store_true")
    cr.add_dataset_args(p)
    cr.add_trials_arg(p)
    return p.parse_args()


def _dims(kind: str, dim: int, ratio: float, seed: int, query: np.ndarray) -> np.ndarray:
    if kind == "random":
        return cr.subspace_indices(dim, ratio, seed=seed)
    if kind == "contiguous":
        return cr.contiguous_indices(dim, ratio)
    return cr.top_energy_indices(query, ratio)


def main() -> int:
    args = parse_args()
    xq = args.xqdrant_url or args.qdrant_url
    run = cr.init_research_run(
        STEP_ID, METRIC, mode=args.mode, dimensions=args.dimensions,
        queries=args.queries, qdrant_url=args.qdrant_url, xqdrant_url=xq,
        sweep={"ratio": args.ratio, "constructions": list(CONSTRUCTIONS)},
        extra={"trials": args.trials, "dataset": args.dataset},
    )
    trial_rows = []
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
        for trial in range(args.trials):
            seed = cr.trial_seed(trial)
            queries, q_idx = cr.select_queries(dataset, args.queries, seed=seed)
            for constr in CONSTRUCTIONS:
                for method in METHODS:
                    lat = LatencyStats()
                    recalls = []
                    for local_i, q in enumerate(queries):
                        dims = _dims(constr, dim, args.ratio, seed, q)
                        gt = cr.full_top_k(dataset, int(q_idx[local_i]), q, args.k, norms)
                        if args.mode == "http":
                            if method == "posthoc":
                                body = cr.focus_body(q, args.k, args.ef_search, dims, masked=False)
                            elif method == "m1":
                                body = cr.focus_body(q, args.k, args.ef_search, dims, masked=True)
                            else:
                                body = cr.focus_body(
                                    q, args.k, args.ef_search, dims,
                                    masked=True, mask_from_layer=args.layer,
                                )
                            out = client.query(body)
                            if out.status != "ok":
                                continue
                            lat.record(out.latency_ns)
                            recalls.append(cr.recall_at_k(out.result.ids, gt, args.k))
                        else:
                            if method == "posthoc":
                                rec = 0.92 if constr != "random" else 0.88
                            elif method == "m1":
                                rec = {"random": 0.08, "contiguous": 0.35, "top_energy": 0.40}[constr]
                            else:
                                rec = {"random": 0.90, "contiguous": 0.93, "top_energy": 0.94}[constr]
                            recalls.append(rec)
                            lat.record(1_200_000 if method != "posthoc" else 900_000)
                    trial_rows.append({
                        "trial": trial, "dimension": dim, "construction": constr,
                        "method": method, "ratio": args.ratio,
                        "recall_at_k": float(np.mean(recalls)) if recalls else float("nan"),
                        "p50_ms": lat.percentiles_ms()["p50"],
                        "simulated": int(args.mode == "simulated"),
                    })
                    print(f"  {constr}/{method} trial={trial}: "
                          f"recall={trial_rows[-1]['recall_at_k']:.3f}")

    rows = cr.write_trial_and_summary_csv(
        cr.bench_common.RESULTS_DIR / "focus_baselines.csv",
        trial_rows, key_fields=["dimension", "construction", "method", "ratio"],
    )
    plt = cr._mpl()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(CONSTRUCTIONS))
    w = 0.25
    for i, method in enumerate(METHODS):
        ys, es = [], []
        for c in CONSTRUCTIONS:
            match = [r for r in rows if r["construction"] == c and r["method"] == method]
            ys.append(float(match[0]["recall_at_k"]) if match else np.nan)
            es.append(float(match[0].get("recall_at_k_std", 0.0) or 0.0) if match else 0.0)
        ax.bar(x + (i - 1) * w, ys, w, yerr=es if any(e > 0 for e in es) else None,
               capsize=3, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(CONSTRUCTIONS)
    ax.set_ylabel("Recall@10 vs full GT")
    prefix = "[SIMULATED] " if args.mode == "simulated" else ""
    ax.set_title(f"{prefix}Post-hoc vs M1/M2 (ratio={args.ratio})")
    ax.legend()
    fig.tight_layout()
    cr.save_fig(fig, "fig13_focus_baselines")
    print(f"[focus_baselines] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
