#!/usr/bin/env python3
"""
Pre-flight HTTP validation for the XQdrant benchmark suite.

Runs a minimal probe against each API surface the benchmarks depend on,
so failures surface in seconds instead of mid-way through a long run.

Usage:
    python validate_http.py
    python validate_http.py --qdrant-url http://127.0.0.1:6335 --xqdrant-url http://127.0.0.1:6333
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

from bench_backends import HttpBackend
from bench_common import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Qdrant/XQdrant HTTP endpoints")
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
    )
    parser.add_argument(
        "--xqdrant-url",
        default=os.environ.get("XQDRANT_URL", os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")),
    )
    parser.add_argument("--dimension", type=int, default=8, help="Small test vector size")
    return parser.parse_args()


def ok(label: str) -> None:
    print(f"  OK  {label}")


def fail(label: str, exc: BaseException) -> None:
    print(f"  FAIL {label}")
    print(f"       {exc}")
    if not isinstance(exc, KeyboardInterrupt):
        traceback.print_exc(limit=2)


def main() -> int:
    args = parse_args()
    print("HTTP pre-flight validation")
    print(f"  Qdrant URL:  {args.qdrant_url}")
    print(f"  XQdrant URL: {args.xqdrant_url}")

    # Tiny dataset keeps provisioning fast.
    dataset = generate_dataset(dimension=args.dimension, num_vectors=32, num_queries=4, seed=99)
    query = dataset.queries[0]
    k = 3
    ef = 64
    focus_dims = np.array([0, 1, 3], dtype=np.int64)

    errors = 0

    def step(name: str, func) -> None:
        nonlocal errors
        try:
            func()
            ok(name)
        except Exception as exc:  # noqa: BLE001 — report all probe failures
            fail(name, exc)
            errors += 1

    vanilla = HttpBackend(
        dataset,
        qdrant_url=args.qdrant_url,
        xqdrant_url=args.qdrant_url,
        collection_name="bench_validate_vanilla",
        setup_collection=True,
    )
    xqdrant = HttpBackend(
        dataset,
        qdrant_url=args.qdrant_url,
        xqdrant_url=args.xqdrant_url,
        collection_name="bench_validate_xqdrant",
        setup_collection=True,
    )

    step("Vanilla Qdrant — nearest query", lambda: vanilla.vanilla_search(query, k, ef))
    step(
        "XQdrant — with_dims_explained",
        lambda: xqdrant.xqdrant_attribution(query, k, m=3, ef_search=ef),
    )
    step(
        "Vanilla Qdrant — post-query vector retrieve",
        lambda: vanilla.post_query_attribution(query, k, m=3, ef_search=ef),
    )
    step(
        "XQdrant — focus rescore (Test D)",
        lambda: xqdrant.vanilla_search(query, k, ef, dim_indices=focus_dims),
    )
    step(
        "XQdrant — masked HNSW focus (Test E)",
        lambda: xqdrant.vanilla_search(
            query, k, ef, dim_indices=focus_dims, focus_masked=True
        ),
    )

    print()
    if errors:
        print(f"Validation failed: {errors} check(s) failed.")
        return 1
    print("All HTTP probes passed. Safe to run ./run_bench.sh in http mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
