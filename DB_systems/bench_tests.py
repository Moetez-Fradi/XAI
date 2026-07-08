"""
Individual benchmark tests (A–D) for the XQdrant evaluation suite.
"""

from __future__ import annotations

import concurrent.futures
from typing import Callable

import numpy as np

from bench_backends import SearchBackend
import bench_common
from bench_common import (
    ATTRIBUTION_DEPTHS,
    DEFAULT_ATTRIBUTION_M,
    DEFAULT_EF_SEARCH,
    EF_SEARCH_VALUES,
    NUM_QUERIES,
    SUBSPACE_RATIOS,
    THREAD_COUNTS,
    TOP_K,
    WARMUP_QUERIES,
    Dataset,
    LatencyStats,
    brute_force_top_k,
    detect_physical_cores,
    recall_at_k,
    write_csv,
)


def _select_queries(dataset: Dataset, num_queries: int) -> np.ndarray:
    return dataset.queries[: min(num_queries, dataset.num_queries)]


def _subspace_indices(dimension: int, ratio: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + int(ratio * 10_000))
    count = max(1, int(round(dimension * ratio)))
    return np.sort(rng.choice(dimension, size=count, replace=False))


# ---------------------------------------------------------------------------
# Test A — Latency vs. Recall@K across ef_search
# ---------------------------------------------------------------------------


def run_test_latency_recall(
    vanilla: SearchBackend,
    xqdrant: SearchBackend,
    dataset: Dataset,
    num_queries: int = NUM_QUERIES,
    k: int = TOP_K,
) -> list[dict]:
    """
    Sweep ef_search logarithmically and compare Vanilla vs XQdrant recall parity
    and latency distributions.
    """
    queries = _select_queries(dataset, num_queries)
    rows: list[dict] = []

    for ef_search in EF_SEARCH_VALUES:
        vanilla.warmup(queries, WARMUP_QUERIES)
        xqdrant.warmup(queries, WARMUP_QUERIES)

        vanilla_stats = LatencyStats()
        xqdrant_stats = LatencyStats()
        vanilla_recalls: list[float] = []
        xqdrant_recalls: list[float] = []

        for query in queries:
            gt_ids, _ = brute_force_top_k(query, dataset.vectors, k)

            v_result, v_ns = vanilla.vanilla_search(query, k, ef_search)
            vanilla_stats.record(v_ns)
            vanilla_recalls.append(recall_at_k(v_result.ids, gt_ids, k))

            x_result, x_ns = xqdrant.xqdrant_attribution(
                query, k, m=DEFAULT_ATTRIBUTION_M, ef_search=ef_search
            )
            xqdrant_stats.record(x_ns)
            xqdrant_recalls.append(recall_at_k(x_result.ids, gt_ids, k))

        v_lat = vanilla_stats.percentiles_ms()
        x_lat = xqdrant_stats.percentiles_ms()
        rows.append(
            {
                "dimension": dataset.dimension,
                "ef_search": ef_search,
                "k": k,
                "vanilla_recall_at_k": float(np.mean(vanilla_recalls)),
                "xqdrant_recall_at_k": float(np.mean(xqdrant_recalls)),
                "vanilla_recall_gap": float(
                    np.mean(vanilla_recalls) - np.mean(xqdrant_recalls)
                ),
                "vanilla_p50_ms": v_lat["p50"],
                "vanilla_p95_ms": v_lat["p95"],
                "vanilla_p99_ms": v_lat["p99"],
                "xqdrant_p50_ms": x_lat["p50"],
                "xqdrant_p95_ms": x_lat["p95"],
                "xqdrant_p99_ms": x_lat["p99"],
            }
        )

    write_csv(
        bench_common.RESULTS_DIR / f"latency_recall_d{dataset.dimension}.csv",
        fieldnames=list(rows[0].keys()),
        rows=rows,
    )
    return rows


# ---------------------------------------------------------------------------
# Test B — Multi-threaded throughput (QPS)
# ---------------------------------------------------------------------------


def run_test_throughput(
    backend: SearchBackend,
    dataset: Dataset,
    num_queries: int = NUM_QUERIES,
    k: int = TOP_K,
    ef_search: int = DEFAULT_EF_SEARCH,
    thread_counts: list[int] | None = None,
) -> list[dict]:
    """
    Saturate the server/client with increasing thread counts and measure QPS.
    """
    queries = _select_queries(dataset, num_queries)
    max_cores = detect_physical_cores()
    counts = thread_counts or [t for t in THREAD_COUNTS if t <= max_cores]
    if not counts:
        counts = [1]

    rows: list[dict] = []

    for threads in counts:
        backend.warmup(queries, WARMUP_QUERIES)

        def _one_query(query: np.ndarray) -> int:
            _, elapsed = backend.vanilla_search(query, k, ef_search)
            return elapsed

        import time as time_module

        start_wall = time_module.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(_one_query, queries))
        wall_s = time_module.perf_counter() - start_wall

        qps = len(queries) / wall_s if wall_s > 0 else 0.0
        rows.append(
            {
                "dimension": dataset.dimension,
                "threads": threads,
                "ef_search": ef_search,
                "num_queries": len(queries),
                "wall_time_s": wall_s,
                "qps": qps,
            }
        )

    write_csv(
        bench_common.RESULTS_DIR / f"throughput_d{dataset.dimension}.csv",
        fieldnames=list(rows[0].keys()),
        rows=rows,
    )
    return rows


# ---------------------------------------------------------------------------
# Test C — Attribution depth (m) scaling overhead
# ---------------------------------------------------------------------------


def run_test_attribution_depth(
    post_query: SearchBackend,
    xqdrant: SearchBackend,
    dataset: Dataset,
    num_queries: int = NUM_QUERIES,
    k: int = TOP_K,
    ef_search: int = DEFAULT_EF_SEARCH,
    depths: list[int] | None = None,
) -> list[dict]:
    """
    Compare Post-Query Extraction vs XQdrant in-database attribution cost
    as the number of requested top features m grows.
    """
    queries = _select_queries(dataset, num_queries)
    m_values = depths or ATTRIBUTION_DEPTHS
    rows: list[dict] = []

    for m in m_values:
        post_query.warmup(queries, WARMUP_QUERIES)
        xqdrant.warmup(queries, WARMUP_QUERIES)

        post_stats = LatencyStats()
        xq_stats = LatencyStats()

        for query in queries:
            _, post_ns = post_query.post_query_attribution(query, k, m, ef_search)
            post_stats.record(post_ns)

            _, xq_ns = xqdrant.xqdrant_attribution(query, k, m, ef_search)
            xq_stats.record(xq_ns)

        post_lat = post_stats.percentiles_ms()
        xq_lat = xq_stats.percentiles_ms()
        rows.append(
            {
                "dimension": dataset.dimension,
                "m": m,
                "k": k,
                "ef_search": ef_search,
                "post_query_p50_ms": post_lat["p50"],
                "post_query_p95_ms": post_lat["p95"],
                "post_query_p99_ms": post_lat["p99"],
                "xqdrant_p50_ms": xq_lat["p50"],
                "xqdrant_p95_ms": xq_lat["p95"],
                "xqdrant_p99_ms": xq_lat["p99"],
                "overhead_p50_ms": post_lat["p50"] - xq_lat["p50"],
                "overhead_p95_ms": post_lat["p95"] - xq_lat["p95"],
                "speedup_p50": (
                    post_lat["p50"] / xq_lat["p50"] if xq_lat["p50"] > 0 else 0.0
                ),
            }
        )

    write_csv(
        bench_common.RESULTS_DIR / f"attribution_depth_d{dataset.dimension}.csv",
        fieldnames=list(rows[0].keys()),
        rows=rows,
    )
    return rows


# ---------------------------------------------------------------------------
# Test D — Dynamic subspace pruning speedup
# ---------------------------------------------------------------------------


def run_test_subspace_pruning(
    backend: SearchBackend,
    dataset: Dataset,
    num_queries: int = NUM_QUERIES,
    k: int = TOP_K,
    ef_search: int = DEFAULT_EF_SEARCH,
    ratios: list[float] | None = None,
) -> list[dict]:
    """
    Test D — Focus *rescore* path (preselect full HNSW + subspace rescore).
    """
    queries = _select_queries(dataset, num_queries)
    ratio_values = ratios or SUBSPACE_RATIOS
    rows: list[dict] = []

    backend.warmup(queries, WARMUP_QUERIES)
    full_stats = LatencyStats()
    for query in queries:
        _, elapsed = backend.vanilla_search(query, k, ef_search)
        full_stats.record(elapsed)
    full_p50 = full_stats.percentiles_ms()["p50"]

    for ratio in ratio_values:
        dim_indices = _subspace_indices(dataset.dimension, ratio)
        backend.warmup(queries, WARMUP_QUERIES)

        sub_stats = LatencyStats()
        for query in queries:
            _, elapsed = backend.vanilla_search(
                query,
                k,
                ef_search,
                dim_indices=dim_indices,
                focus_masked=False,
            )
            sub_stats.record(elapsed)

        sub_p50 = sub_stats.percentiles_ms()["p50"]
        speedup = full_p50 / sub_p50 if sub_p50 > 0 else 0.0
        rows.append(
            {
                "dimension": dataset.dimension,
                "mode": "rescore",
                "subspace_ratio": ratio,
                "subspace_dims": len(dim_indices),
                "ef_search": ef_search,
                "full_p50_ms": full_p50,
                "subspace_p50_ms": sub_p50,
                "subspace_p95_ms": sub_stats.percentiles_ms()["p95"],
                "speedup_factor": speedup,
            }
        )

    rows.insert(
        0,
        {
            "dimension": dataset.dimension,
            "mode": "rescore",
            "subspace_ratio": 1.0,
            "subspace_dims": dataset.dimension,
            "ef_search": ef_search,
            "full_p50_ms": full_p50,
            "subspace_p50_ms": full_p50,
            "subspace_p95_ms": full_stats.percentiles_ms()["p95"],
            "speedup_factor": 1.0,
        },
    )

    write_csv(
        bench_common.RESULTS_DIR / f"subspace_rescore_d{dataset.dimension}.csv",
        fieldnames=list(rows[0].keys()),
        rows=rows,
    )
    # Legacy filename for older plot scripts / papers
    write_csv(
        bench_common.RESULTS_DIR / f"subspace_pruning_d{dataset.dimension}.csv",
        fieldnames=list(rows[0].keys()),
        rows=rows,
    )
    return rows


# ---------------------------------------------------------------------------
# Test E — Masked HNSW subspace search (focus.masked = true)
# ---------------------------------------------------------------------------


def run_test_masked_subspace(
    backend: SearchBackend,
    dataset: Dataset,
    num_queries: int = NUM_QUERIES,
    k: int = TOP_K,
    ef_search: int = DEFAULT_EF_SEARCH,
    ratios: list[float] | None = None,
) -> list[dict]:
    """
    Test E — Compare focus rescore vs masked HNSW traversal.

    Measures latency speedup and recall@K against exact subspace ground truth.
    """
    queries = _select_queries(dataset, num_queries)
    ratio_values = ratios or SUBSPACE_RATIOS
    rows: list[dict] = []

    backend.warmup(queries, WARMUP_QUERIES)
    full_stats = LatencyStats()
    for query in queries:
        _, elapsed = backend.vanilla_search(query, k, ef_search)
        full_stats.record(elapsed)
    full_p50 = full_stats.percentiles_ms()["p50"]

    for ratio in ratio_values:
        dim_indices = _subspace_indices(dataset.dimension, ratio)
        backend.warmup(queries, WARMUP_QUERIES)

        rescore_stats = LatencyStats()
        masked_stats = LatencyStats()
        rescore_recalls: list[float] = []
        masked_recalls: list[float] = []

        for query in queries:
            gt_ids, _ = brute_force_top_k(
                query, dataset.vectors, k, dim_indices=dim_indices
            )

            res_result, res_ns = backend.vanilla_search(
                query,
                k,
                ef_search,
                dim_indices=dim_indices,
                focus_masked=False,
            )
            rescore_stats.record(res_ns)
            rescore_recalls.append(
                recall_at_k(res_result.ids, gt_ids, k)
            )

            masked_result, masked_ns = backend.vanilla_search(
                query,
                k,
                ef_search,
                dim_indices=dim_indices,
                focus_masked=True,
            )
            masked_stats.record(masked_ns)
            masked_recalls.append(
                recall_at_k(masked_result.ids, gt_ids, k)
            )

        rescore_p50 = rescore_stats.percentiles_ms()["p50"]
        masked_p50 = masked_stats.percentiles_ms()["p50"]
        rows.append(
            {
                "dimension": dataset.dimension,
                "subspace_ratio": ratio,
                "subspace_dims": len(dim_indices),
                "ef_search": ef_search,
                "full_p50_ms": full_p50,
                "rescore_p50_ms": rescore_p50,
                "masked_p50_ms": masked_p50,
                "rescore_speedup": full_p50 / rescore_p50 if rescore_p50 > 0 else 0.0,
                "masked_speedup": full_p50 / masked_p50 if masked_p50 > 0 else 0.0,
                "rescore_recall_at_k": float(np.mean(rescore_recalls)),
                "masked_recall_at_k": float(np.mean(masked_recalls)),
            }
        )

    rows.insert(
        0,
        {
            "dimension": dataset.dimension,
            "subspace_ratio": 1.0,
            "subspace_dims": dataset.dimension,
            "ef_search": ef_search,
            "full_p50_ms": full_p50,
            "rescore_p50_ms": full_p50,
            "masked_p50_ms": full_p50,
            "rescore_speedup": 1.0,
            "masked_speedup": 1.0,
            "rescore_recall_at_k": 1.0,
            "masked_recall_at_k": 1.0,
        },
    )

    write_csv(
        bench_common.RESULTS_DIR / f"masked_subspace_d{dataset.dimension}.csv",
        fieldnames=list(rows[0].keys()),
        rows=rows,
    )
    return rows
