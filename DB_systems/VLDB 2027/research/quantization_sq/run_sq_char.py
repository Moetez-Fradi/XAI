#!/usr/bin/env python3
"""Fig 21 — scalar quantization characterization.

Measures: (i) plain HNSW Recall@10 FP vs SQ int8, (ii) masked-focus fail-fast
on the SQ collection, (iii) attribution sum-consistency (sum of per-dim terms
vs engine score / L2^2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import DEFAULT_EF_SEARCH, TOP_K, score_contributions  # noqa: E402

STEP_ID = "quantization_sq"
METRIC = "sq_char"


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
    p.add_argument("--no-provision", action="store_true")
    cr.add_dataset_args(p)
    cr.add_trials_arg(p)
    return p.parse_args()


def _sum_rel_error(query, vector, distance, terms: dict[int, float], score: float) -> float:
    contrib = score_contributions(query, vector, distance)
    s = float(np.sum(contrib))
    # For Euclid, engine score is typically -L2^2 or L2; compare to sum of (q-v)^2.
    denom = abs(s) + 1e-12
    return abs(s - float(score)) / denom if np.isfinite(score) else abs(s - sum(terms.values())) / denom


def main() -> int:
    args = parse_args()
    xq = args.xqdrant_url or args.qdrant_url
    run = cr.init_research_run(
        STEP_ID, METRIC, mode=args.mode, dimensions=args.dimensions,
        queries=args.queries, qdrant_url=args.qdrant_url, xqdrant_url=xq,
        extra={"trials": args.trials},
    )
    trial_rows = []
    for dataset in cr.iter_datasets(args):
        dim = dataset.dimension
        norms = cr.full_norms_sq(dataset)
        fp_client = sq_client = None
        if args.mode == "http":
            fp_client = cr.ResearchClient(
                dataset, args.qdrant_url, xq,
                provision=not args.no_provision,
                collection_name=f"sq_fp_{dataset.name}",
            )
            sq_client = cr.ResearchClient(
                dataset, args.qdrant_url, xq,
                provision=not args.no_provision,
                quantization="sq",
                collection_name=f"sq_int8_{dataset.name}",
            )
            if not args.no_provision:
                fp_client.wait_for_indexing(expected_points=dataset.num_vectors)
                sq_client.wait_for_indexing(expected_points=dataset.num_vectors)

        for trial in range(args.trials):
            seed = cr.trial_seed(trial)
            queries, q_idx = cr.select_queries(dataset, args.queries, seed=seed)
            dims = cr.subspace_indices(dim, args.ratio, seed=seed)
            fp_rec, sq_rec, m2_rec = [], [], []
            masked_rejected = 0
            masked_tried = 0
            fp_err, sq_err = [], []
            for local_i, q in enumerate(queries):
                gt = cr.full_top_k(dataset, int(q_idx[local_i]), q, args.k, norms)
                if args.mode == "http":
                    fp_out = fp_client.query(cr.focus_body(q, args.k, args.ef_search))
                    sq_out = sq_client.query(cr.focus_body(q, args.k, args.ef_search))
                    if fp_out.status == "ok":
                        fp_rec.append(cr.recall_at_k(fp_out.result.ids, gt, args.k))
                    if sq_out.status == "ok":
                        sq_rec.append(cr.recall_at_k(sq_out.result.ids, gt, args.k))
                    m2 = fp_client.query(cr.focus_body(
                        q, args.k, args.ef_search, dims, masked=True, mask_from_layer=1
                    ))
                    if m2.status == "ok":
                        m2_rec.append(cr.recall_at_k(m2.result.ids, gt, args.k))
                    masked_tried += 1
                    sq_mask = sq_client.query(cr.focus_body(
                        q, args.k, args.ef_search, dims, masked=True, mask_from_layer=1
                    ))
                    if sq_mask.status != "ok":
                        masked_rejected += 1
                    # Attribution sum-consistency on first hit
                    attr_body = cr.focus_body(q, args.k, args.ef_search)
                    attr_body["with_dims_explained"] = {"top": dim}
                    fp_a = fp_client.query(attr_body)
                    sq_a = sq_client.query(attr_body)
                    row = int(q_idx[local_i]) if False else None  # noqa: F841
                    if fp_a.status == "ok" and fp_a.result and fp_a.result.dims_explained:
                        terms = fp_a.result.dims_explained[0]
                        fp_err.append(abs(sum(terms.values()) - fp_a.result.scores[0]) /
                                      (abs(fp_a.result.scores[0]) + 1e-12))
                    if sq_a.status == "ok" and sq_a.result and sq_a.result.dims_explained:
                        terms = sq_a.result.dims_explained[0]
                        sq_err.append(abs(sum(terms.values()) - sq_a.result.scores[0]) /
                                      (abs(sq_a.result.scores[0]) + 1e-12))
                else:
                    fp_rec.append(1.0)
                    sq_rec.append(0.982)
                    m2_rec.append(0.75)
                    masked_tried += 1
                    masked_rejected += 1
                    fp_err.append(1e-6)
                    sq_err.append(0.0059)

            trial_rows.append({
                "trial": trial, "dimension": dim, "ratio": args.ratio,
                "fp_recall": float(np.mean(fp_rec)) if fp_rec else float("nan"),
                "sq_recall": float(np.mean(sq_rec)) if sq_rec else float("nan"),
                "fp_m2_recall": float(np.mean(m2_rec)) if m2_rec else float("nan"),
                "sq_mask_reject_rate": masked_rejected / max(1, masked_tried),
                "fp_attr_rel_err": float(np.mean(fp_err)) if fp_err else float("nan"),
                "sq_attr_rel_err": float(np.mean(sq_err)) if sq_err else float("nan"),
                "simulated": int(args.mode == "simulated"),
            })
            print(f"  trial={trial}: FP={trial_rows[-1]['fp_recall']:.3f} "
                  f"SQ={trial_rows[-1]['sq_recall']:.3f} "
                  f"reject={trial_rows[-1]['sq_mask_reject_rate']:.2f}")

    rows = cr.write_trial_and_summary_csv(
        cr.bench_common.RESULTS_DIR / "sq_char.csv",
        trial_rows, key_fields=["dimension", "ratio"],
    )
    r = rows[0]
    plt = cr._mpl()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    axes[0].bar(["FP", "SQ int8"], [float(r["fp_recall"]), float(r["sq_recall"])],
                color=["#4c72b0", "#c44e52"])
    axes[0].set_ylim(0.9, 1.01)
    axes[0].set_ylabel("Plain Recall@10")
    axes[0].set_title("Plain search")
    axes[1].bar(["FP M2", "SQ+mask"], [float(r["fp_m2_recall"]),
                                       0.0 if float(r["sq_mask_reject_rate"]) > 0.5 else float("nan")],
                color=["#55a868", "#dd8452"])
    axes[1].set_title("Masked path")
    axes[1].set_ylabel("Recall / reject")
    axes[2].bar(["FP", "SQ"], [float(r["fp_attr_rel_err"]) * 100,
                               float(r["sq_attr_rel_err"]) * 100],
                color=["#4c72b0", "#c44e52"])
    axes[2].set_ylabel("Rel. error (%)")
    axes[2].set_title("Attribution sum")
    prefix = "[SIMULATED] " if args.mode == "simulated" else ""
    fig.suptitle(f"{prefix}SQ characterization", y=1.03)
    fig.tight_layout()
    cr.save_fig(fig, "fig21_sq_char")
    print(f"[quantization_sq] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
