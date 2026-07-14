#!/usr/bin/env python3
"""
Cross-cut X3 — Fallback / verify pass (corrected evaluation).

Requires XQdrant ``focus.verify``. Compares masked vs masked+verify under:

1. **Full-space GT (primary)** — does verify recover *full-vector* recall?
2. **Subspace GT (secondary)** — for reference; full re-rank often *hurts* this.
3. **Overfetch** — request ``limit = m·k`` with verify, keep top-k (true fast-filter).
4. **Hybrid** — optional ``mask_from_layer`` + verify.

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
DEFAULT_OVERFETCH = [1, 8]
# -1 means fully masked (omit mask_from_layer); positive = hybrid cutoff
DEFAULT_LAYERS = [-1, 1]


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
    p.add_argument(
        "--overfetch",
        type=int,
        nargs="+",
        default=DEFAULT_OVERFETCH,
        help="Candidate multipliers m: query limit=m*k with verify, evaluate top-k "
             "(m=1 is plain verify). Example: --overfetch 1 4 16",
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="mask_from_layer values; use -1 for fully masked (omit field). "
             "Example: --layers -1 1 2",
    )
    p.add_argument("--no-provision", action="store_true")
    p.add_argument("--experiment-id", default=None,
                   help="Optional run_id suffix override for experiments/ folder name")
    cr.add_dataset_args(p)
    return p.parse_args()


def _layer_tag(layer: int) -> str | int:
    return "all_masked" if layer < 0 else layer


def _mask_from_layer_arg(layer: int) -> int | None:
    return None if layer < 0 else int(layer)


def _run_http_variant(
    client,
    queries,
    dims,
    k: int,
    ef: int,
    *,
    verify: bool,
    limit: int,
    mask_from_layer: int | None,
) -> tuple[LatencyStats, list[list[int]], int]:
    """Return latency, per-query id lists (length <= limit), unsupported count."""
    lat = LatencyStats()
    id_lists: list[list[int]] = []
    unsupported = 0
    for q in queries:
        body = cr.focus_body(
            q, limit, ef, dims,
            masked=True,
            mask_from_layer=mask_from_layer,
            verify=True if verify else None,
        )
        out = client.query(body)
        if out.status != "ok" or out.result is None:
            unsupported += 1
            id_lists.append([])
            continue
        lat.record(out.latency_ns)
        id_lists.append(list(out.result.ids))
    return lat, id_lists, unsupported


def _run_simulated_variant(
    dataset,
    queries,
    dims,
    k: int,
    *,
    verify: bool,
    overfetch: int,
    mask_from_layer: int | None,
    norms_sq,
) -> tuple[LatencyStats, list[list[int]], int]:
    """Offline stand-in: masked ≈ subspace top; verify ≈ blend toward full GT."""
    lat = LatencyStats()
    id_lists: list[list[int]] = []
    for i, q in enumerate(queries):
        res, ns, _ = cr.simulated_masked(
            q, dataset, k, dims, mask_from_layer=mask_from_layer,
        )
        ids = list(res.ids)
        if verify:
            full_ids = cr.full_top_k(dataset, i, q, k, norms_sq)
            # Overfetch models a larger candidate pool before full re-rank.
            pool = max(k, overfetch * k)
            sub_ids, _ = cr.SubspaceGT(
                dataset.vectors, dims, distance=dataset.distance,
            ).top_k(q, pool)
            # Prefer full-GT hits that appear in the subspace pool.
            sub_set = set(int(x) for x in sub_ids.tolist())
            ranked = [int(x) for x in full_ids.tolist() if int(x) in sub_set]
            filler = [int(x) for x in sub_ids.tolist() if int(x) not in set(ranked)]
            ids = (ranked + filler)[:k]
            ns = int(ns * (1.0 + 0.05 * overfetch))
        lat.record(ns)
        id_lists.append(ids[:k])
    return lat, id_lists, 0


def _mean_recall(id_lists: list[list[int]], gt_list: list[np.ndarray], k: int) -> float:
    vals = [
        cr.recall_at_k(ids[:k], gt, k)
        for ids, gt in zip(id_lists, gt_list)
        if ids
    ]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    args = parse_args()
    for m in args.overfetch:
        if m < 1:
            raise SystemExit("--overfetch values must be >= 1")

    xqdrant_url = args.xqdrant_url or args.qdrant_url
    run_id = None
    if args.experiment_id:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_id = f"{ts}__{args.experiment_id}"

    run = cr.init_research_run(
        STEP_ID, METRIC,
        mode=args.mode, dimensions=args.dimensions, queries=args.queries,
        qdrant_url=args.qdrant_url, xqdrant_url=xqdrant_url,
        requires_xqdrant_change="focus.verify (final full-distance rescore of top-k)",
        sweep={
            "ratios": args.ratios,
            "ef_search": args.ef_search,
            "k": args.k,
            "overfetch": args.overfetch,
            "layers": args.layers,
        },
        extra={"dataset": args.dataset},
        run_id=run_id,
    )
    print(f"[xcut3] experiment: {run.root}")
    print(f"[xcut3] primary metric = full-space Recall@{args.k}; "
          f"overfetch={args.overfetch}; layers={[_layer_tag(L) for L in args.layers]}")

    for dataset in cr.iter_datasets(args):
        dim = dataset.dimension
        queries = dataset.queries[: min(args.queries, dataset.num_queries)]
        norms = cr.full_norms_sq(dataset)
        full_gt = [cr.full_top_k(dataset, i, q, args.k, norms)
                   for i, q in enumerate(queries)]

        client = None
        if args.mode == "http":
            client = cr.ResearchClient(
                dataset, args.qdrant_url, xqdrant_url, provision=not args.no_provision,
            )
            if not args.no_provision:
                print(f"  Waiting for index build (D={dim}, N={dataset.num_vectors})...")
                client.wait_for_indexing(expected_points=dataset.num_vectors)
            client.warmup(queries, min(200, len(queries)))

        rows: list[dict] = []
        # For plotting: one series set per layer, full-recall vs ratio
        plot_by_layer: dict[str, dict[str, list[float]]] = {}

        for layer in args.layers:
            mfl = _mask_from_layer_arg(layer)
            layer_label = str(_layer_tag(layer))
            plot_by_layer[layer_label] = {
                "ratios": [],
                "masked_full": [],
            }
            for m in args.overfetch:
                plot_by_layer[layer_label][f"verify_m{m}_full"] = []

            for ratio in args.ratios:
                dims = cr.subspace_indices(dim, ratio)
                subspace_gt = cr.SubspaceGT(
                    dataset.vectors, dims, distance=dataset.distance,
                )
                sub_gt = [subspace_gt.top_k(q, args.k)[0] for q in queries]

                # Masked baseline (limit=k, verify off) — once per ratio/layer
                if args.mode == "http":
                    off_lat, off_ids, off_un = _run_http_variant(
                        client, queries, dims, args.k, args.ef_search,
                        verify=False, limit=args.k, mask_from_layer=mfl,
                    )
                else:
                    off_lat, off_ids, off_un = _run_simulated_variant(
                        dataset, queries, dims, args.k,
                        verify=False, overfetch=1, mask_from_layer=mfl, norms_sq=norms,
                    )

                masked_full = _mean_recall(off_ids, full_gt, args.k)
                masked_sub = _mean_recall(off_ids, sub_gt, args.k)
                off_p50 = off_lat.percentiles_ms()["p50"]

                plot_by_layer[layer_label]["ratios"].append(ratio)
                plot_by_layer[layer_label]["masked_full"].append(masked_full)

                for m in args.overfetch:
                    limit = m * args.k
                    if args.mode == "http":
                        on_lat, on_ids, on_un = _run_http_variant(
                            client, queries, dims, args.k, args.ef_search,
                            verify=True, limit=limit, mask_from_layer=mfl,
                        )
                    else:
                        on_lat, on_ids, on_un = _run_simulated_variant(
                            dataset, queries, dims, args.k,
                            verify=True, overfetch=m, mask_from_layer=mfl,
                            norms_sq=norms,
                        )

                    # Keep top-k after server returned up to limit (already full-ranked if verify)
                    truncated = [ids[: args.k] for ids in on_ids]
                    verify_full = _mean_recall(truncated, full_gt, args.k)
                    verify_sub = _mean_recall(truncated, sub_gt, args.k)
                    on_p50 = on_lat.percentiles_ms()["p50"]

                    row = {
                        "dimension": dim,
                        "subspace_ratio": ratio,
                        "subspace_dims": len(dims),
                        "mask_from_layer": layer_label,
                        "overfetch": m,
                        "query_limit": limit,
                        "ef_search": args.ef_search,
                        "k": args.k,
                        "masked_full_recall": masked_full,
                        "verify_full_recall": verify_full,
                        "full_recall_recovered": (
                            verify_full - masked_full
                            if (not np.isnan(verify_full) and not np.isnan(masked_full))
                            else float("nan")
                        ),
                        "masked_subspace_recall": masked_sub,
                        "verify_subspace_recall": verify_sub,
                        "subspace_recall_delta": (
                            verify_sub - masked_sub
                            if (not np.isnan(verify_sub) and not np.isnan(masked_sub))
                            else float("nan")
                        ),
                        "masked_p50_ms": off_p50,
                        "verify_p50_ms": on_p50,
                        "verify_latency_overhead_ms": on_p50 - off_p50,
                        "unsupported_queries": off_un + on_un,
                        "simulated": int(args.mode == "simulated"),
                    }
                    rows.append(row)
                    plot_by_layer[layer_label][f"verify_m{m}_full"].append(verify_full)

                    tag = " [SIM]" if args.mode == "simulated" else ""
                    print(
                        f"  D={dim} layer={layer_label} ratio={ratio} m={m}: "
                        f"full {masked_full:.3f}->{verify_full:.3f} "
                        f"(+{row['full_recall_recovered']:.3f}) "
                        f"sub {masked_sub:.3f}->{verify_sub:.3f} "
                        f"cost=+{row['verify_latency_overhead_ms']:.3f}ms{tag}"
                    )

        if not rows:
            print(f"  D={dim}: no rows")
            continue

        csv_path = cr.bench_common.RESULTS_DIR / f"xcut3_verify_d{dim}.csv"
        cr.write_csv(csv_path, list(rows[0].keys()), rows)

        prefix = "[SIMULATED] " if args.mode == "simulated" else ""
        for layer_label, series_data in plot_by_layer.items():
            xs = series_data["ratios"]
            series = {"masked (full GT)": series_data["masked_full"]}
            for m in args.overfetch:
                label = f"verify m={m} (full GT)" if m > 1 else "verify m=1 (full GT)"
                series[label] = series_data[f"verify_m{m}_full"]
            cr.line_plot(
                xs, series,
                xlabel="Subspace ratio (D_sub / D)",
                ylabel=f"Recall@{args.k} (full-space GT)",
                title=f"{prefix}X3 verify full-space recall (D={dim}, layer={layer_label})",
                stem=f"xcut3_verify_full_recall_d{dim}_layer_{layer_label}",
                hline=None,
            )

            # Secondary: subspace recall for m=1 only (document the old pitfall)
            sub_rows = [
                r for r in rows
                if r["mask_from_layer"] == layer_label and r["overfetch"] == 1
            ]
            if sub_rows:
                cr.line_plot(
                    [r["subspace_ratio"] for r in sub_rows],
                    {
                        "masked (subspace GT)": [r["masked_subspace_recall"] for r in sub_rows],
                        "verify m=1 (subspace GT)": [r["verify_subspace_recall"] for r in sub_rows],
                    },
                    xlabel="Subspace ratio (D_sub / D)",
                    ylabel=f"Recall@{args.k} (subspace GT)",
                    title=f"{prefix}X3 subspace GT (ref only; D={dim}, layer={layer_label})",
                    stem=f"xcut3_verify_subspace_recall_d{dim}_layer_{layer_label}",
                    hline=None,
                )

    print(f"[xcut3] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
