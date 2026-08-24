#!/usr/bin/env python3
"""Fig 12 — M2 scale × ef_search sensitivity on SIFT subsamples.

Sweeps N in {10k, 50k, 100k} and ef in {64, 128, 256} at fixed random focus
ratio 0.5, cutoff L=1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import LatencyStats, TOP_K  # noqa: E402

STEP_ID = "scale_ef"
METRIC = "m2_n_ef"
DEFAULT_NS = [10_000, 50_000, 100_000]
DEFAULT_EFS = [64, 128, 256]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["http", "simulated"], default="http")
    p.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    p.add_argument("--xqdrant-url", default=None)
    p.add_argument("--queries", type=int, default=40)
    p.add_argument("--k", type=int, default=TOP_K)
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS)
    p.add_argument("--efs", type=int, nargs="+", default=DEFAULT_EFS)
    p.add_argument("--dimensions", type=int, nargs="+", default=[128])
    p.add_argument("--no-provision", action="store_true")
    cr.add_dataset_args(p)
    cr.add_trials_arg(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.dataset = args.dataset if args.dataset != "synthetic" else "sift1m"
    xq = args.xqdrant_url or args.qdrant_url
    run = cr.init_research_run(
        STEP_ID, METRIC, mode=args.mode, dimensions=[128], queries=args.queries,
        qdrant_url=args.qdrant_url, xqdrant_url=xq,
        sweep={"ns": args.ns, "efs": args.efs, "ratio": args.ratio},
        extra={"trials": args.trials},
    )
    trial_rows = []
    for n in args.ns:
        args.num_vectors = n
        for dataset in cr.iter_datasets(args):
            dim = dataset.dimension
            norms = cr.full_norms_sq(dataset)
            client = None
            if args.mode == "http":
                client = cr.ResearchClient(
                    dataset, args.qdrant_url, xq,
                    provision=not args.no_provision,
                    collection_name=f"scale_ef_n{n}",
                )
                if not args.no_provision:
                    client.wait_for_indexing(expected_points=dataset.num_vectors)
            for trial in range(args.trials):
                seed = cr.trial_seed(trial)
                queries, q_idx = cr.select_queries(dataset, args.queries, seed=seed)
                dims = cr.subspace_indices(dim, args.ratio, seed=seed)
                for ef in args.efs:
                    lat = LatencyStats()
                    recalls = []
                    for local_i, q in enumerate(queries):
                        gt = cr.full_top_k(dataset, int(q_idx[local_i]), q, args.k, norms)
                        if args.mode == "http":
                            out = client.query(cr.focus_body(
                                q, args.k, ef, dims, masked=True, mask_from_layer=args.layer
                            ))
                            if out.status != "ok":
                                continue
                            lat.record(out.latency_ns)
                            recalls.append(cr.recall_at_k(out.result.ids, gt, args.k))
                        else:
                            # Mild scale penalty + flat-in-ef (paper: variation < 0.01).
                            base = 0.45 - 0.04 * np.log10(max(n, 10) / 10_000)
                            rec = float(np.clip(base + 0.005 * (ef / 128 - 1), 0.05, 0.99))
                            recalls.append(rec)
                            lat.record(int((0.55 + 0.25 * np.log10(n / 10_000)) * 1e6))
                    trial_rows.append({
                        "trial": trial, "n": n, "ef_search": ef, "ratio": args.ratio,
                        "recall_at_k": float(np.mean(recalls)) if recalls else float("nan"),
                        "p50_ms": lat.percentiles_ms()["p50"],
                        "simulated": int(args.mode == "simulated"),
                    })
                    print(f"  n={n} ef={ef} trial={trial}: "
                          f"recall={trial_rows[-1]['recall_at_k']:.3f} "
                          f"p50={trial_rows[-1]['p50_ms']:.3f}ms")

    rows = cr.write_trial_and_summary_csv(
        cr.bench_common.RESULTS_DIR / "scale_ef.csv",
        trial_rows, key_fields=["n", "ef_search", "ratio"],
    )
    plt = cr._mpl()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ef in args.efs:
        xs, rec, rec_e, lat, lat_e = [], [], [], [], []
        for n in args.ns:
            match = [r for r in rows if int(r["n"]) == n and int(r["ef_search"]) == ef]
            if not match:
                continue
            xs.append(n)
            rec.append(float(match[0]["recall_at_k"]))
            rec_e.append(float(match[0].get("recall_at_k_std", 0.0) or 0.0))
            lat.append(float(match[0]["p50_ms"]))
            lat_e.append(float(match[0].get("p50_ms_std", 0.0) or 0.0))
        axes[0].errorbar(xs, rec, yerr=rec_e if any(e > 0 for e in rec_e) else None,
                         marker="o", label=f"ef={ef}")
        axes[1].errorbar(xs, lat, yerr=lat_e if any(e > 0 for e in lat_e) else None,
                         marker="o", label=f"ef={ef}")
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("Recall@10")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("p50 ms")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    prefix = "[SIMULATED] " if args.mode == "simulated" else ""
    fig.suptitle(f"{prefix}M2 N×ef (ratio={args.ratio})", y=1.02)
    fig.tight_layout()
    cr.save_fig(fig, "fig12_scale_ef")
    print(f"[scale_ef] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
