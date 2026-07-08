#!/usr/bin/env python3
"""
Automated Systems Benchmarking and Visualization Suite for XQdrant Evaluation.

Usage (simulated, no server required):
    python bench_suite.py --mode simulated

Usage (live HTTP against running Qdrant / XQdrant):
    python bench_suite.py --mode http \\
        --qdrant-url http://127.0.0.1:6333 \\
        --xqdrant-url http://127.0.0.1:6334

Environment variables:
    QDRANT_URL, XQDRANT_URL, BENCH_MODE, BENCH_DIMENSIONS, BENCH_QUERIES

Outputs:
    ./experiments/<timestamp>/results/*.csv
    ./experiments/<timestamp>/plots/*.png|pdf
    ./experiments/<timestamp>/manifest.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from bench_backends import make_backend
import bench_common
from bench_common import (
    DIMENSIONS,
    NUM_QUERIES,
    ensure_output_dirs,
    generate_dataset,
    init_experiment_run,
)
from bench_tests import (
    run_test_attribution_depth,
    run_test_latency_recall,
    run_test_masked_subspace,
    run_test_subspace_pruning,
    run_test_throughput,
)
from bench_viz import generate_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XQdrant automated benchmarking and visualization suite",
    )
    parser.add_argument(
        "--mode",
        choices=["simulated", "http"],
        default=os.environ.get("BENCH_MODE", "simulated"),
        help="Backend mode: in-memory simulation or live HTTP (default: simulated)",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
        help="Vanilla Qdrant REST base URL",
    )
    parser.add_argument(
        "--xqdrant-url",
        default=os.environ.get("XQDRANT_URL", os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")),
        help="XQdrant REST base URL (defaults to --qdrant-url)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[
            int(x)
            for x in os.environ.get("BENCH_DIMENSIONS", " ".join(str(d) for d in DIMENSIONS)).split()
        ],
        help="Vector dimensions to benchmark (default: 768 1536)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=int(os.environ.get("BENCH_QUERIES", str(NUM_QUERIES))),
        help=f"Number of timed queries per configuration (default: {NUM_QUERIES})",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only write CSV results, skip chart generation",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["A", "B", "C", "D", "E", "all"],
        default=["all"],
        help="Subset of tests to run (default: all)",
    )
    parser.add_argument(
        "--experiment-id",
        default=os.environ.get("BENCH_EXPERIMENT_ID"),
        help="Optional fixed experiment folder name (default: auto timestamp)",
    )
    parser.add_argument(
        "--no-provision",
        action="store_true",
        help="HTTP mode only: skip collection create/upsert (reuse existing data)",
    )
    parser.add_argument(
        "--validate-http",
        action="store_true",
        help="HTTP mode only: run endpoint pre-flight checks and exit",
    )
    return parser.parse_args()


def should_run(tests: list[str], letter: str) -> bool:
    return "all" in tests or letter in tests


def main() -> int:
    args = parse_args()

    if args.validate_http:
        if args.mode != "http":
            print("ERROR: --validate-http requires --mode http", file=sys.stderr)
            return 2
        from validate_http import main as validate_main
        import sys as _sys

        _sys.argv = [
            "validate_http.py",
            "--qdrant-url",
            args.qdrant_url,
            "--xqdrant-url",
            args.xqdrant_url,
        ]
        return validate_main()

    run = init_experiment_run(
        run_id=args.experiment_id,
        metadata={
            "mode": args.mode,
            "qdrant_url": args.qdrant_url,
            "xqdrant_url": args.xqdrant_url,
            "dimensions": args.dimensions,
            "queries": args.queries,
            "tests": args.tests,
        },
    )
    ensure_output_dirs()

    print("=" * 72)
    print("XQdrant Benchmark Suite")
    print("=" * 72)
    print(f"Experiment:  {run.root}")
    print(f"Mode:        {args.mode}")
    print(f"Dimensions:  {args.dimensions}")
    print(f"Queries:     {args.queries}")
    print(f"Results dir: {bench_common.RESULTS_DIR}")
    print(f"Plots dir:   {bench_common.PLOTS_DIR}")
    if args.mode == "http":
        print(f"Qdrant URL:  {args.qdrant_url}")
        print(f"XQdrant URL: {args.xqdrant_url}")
    print("=" * 72)

    evaluated_dims: list[int] = []
    suite_start = time.perf_counter()

    for dimension in args.dimensions:
        print(f"\n--- Dimension D={dimension} ---")
        dataset = generate_dataset(dimension=dimension)
        print(
            f"Dataset: {dataset.num_vectors} vectors, "
            f"{dataset.num_queries} queries, seed={dataset.name}"
        )

        setup = not args.no_provision
        vanilla = make_backend(
            args.mode,
            dataset,
            qdrant_url=args.qdrant_url,
            xqdrant_url=args.qdrant_url,
            setup_collection=setup,
        )
        xqdrant = make_backend(
            args.mode,
            dataset,
            qdrant_url=args.qdrant_url,
            xqdrant_url=args.xqdrant_url,
            setup_collection=setup,
        )
        post_query = vanilla  # post-query baseline uses vanilla URL for search+retrieve

        if should_run(args.tests, "A"):
            print("[Test A] Latency vs. Recall@K (ef_search sweep)...")
            run_test_latency_recall(
                vanilla=vanilla,
                xqdrant=xqdrant,
                dataset=dataset,
                num_queries=args.queries,
            )

        if should_run(args.tests, "B"):
            print("[Test B] Multi-threaded throughput (QPS)...")
            run_test_throughput(
                backend=vanilla,
                dataset=dataset,
                num_queries=args.queries,
            )

        if should_run(args.tests, "C"):
            print("[Test C] Attribution depth (m) scaling...")
            run_test_attribution_depth(
                post_query=post_query,
                xqdrant=xqdrant,
                dataset=dataset,
                num_queries=args.queries,
            )

        if should_run(args.tests, "D"):
            print("[Test D] Focus rescore subspace latency...")
            run_test_subspace_pruning(
                backend=xqdrant,
                dataset=dataset,
                num_queries=args.queries,
            )

        if should_run(args.tests, "E"):
            print("[Test E] Masked HNSW subspace (focus.masked)...")
            run_test_masked_subspace(
                backend=xqdrant,
                dataset=dataset,
                num_queries=args.queries,
            )

        evaluated_dims.append(dimension)

    if not args.skip_plots:
        print("\n[Plots] Generating publication charts...")
        created = generate_all_plots(evaluated_dims)
        for path in created:
            print(f"  wrote {path}")

    elapsed = time.perf_counter() - suite_start
    print(f"\nBenchmark suite completed in {elapsed:.1f}s")
    print(f"Experiment:  {run.root}")
    print(f"CSV results: {bench_common.RESULTS_DIR}")
    print(f"Charts:      {bench_common.PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
