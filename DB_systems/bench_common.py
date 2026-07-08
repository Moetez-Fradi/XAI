"""
Shared constants, data generation, timing utilities, and CSV helpers
for the XQdrant benchmarking suite.
"""

from __future__ import annotations

import csv
import heapq
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
NUM_VECTORS = 10_000
NUM_QUERIES = 500
TOP_K = 10
WARMUP_QUERIES = 1_000

DIMENSIONS = (768, 1536)
EF_SEARCH_VALUES = [16, 22, 32, 45, 64, 90, 128, 181, 256, 362, 512]
THREAD_COUNTS = [1, 2, 4, 8, 16]
ATTRIBUTION_DEPTHS = [1, 5, 10, 20, 50]
SUBSPACE_RATIOS = [0.25, 0.5, 0.75]

DEFAULT_DISTANCE = "Dot"
DEFAULT_EF_SEARCH = 128
DEFAULT_ATTRIBUTION_M = 10

PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = PACKAGE_DIR / "experiments"

# Set per run via init_experiment_run(); legacy paths kept for imports.
RESULTS_DIR = PACKAGE_DIR / "results"
PLOTS_DIR = PACKAGE_DIR / "plots"


@dataclass(frozen=True)
class ExperimentRun:
    """One timestamped benchmark experiment directory."""

    run_id: str
    root: Path
    results_dir: Path
    plots_dir: Path


_active_run: ExperimentRun | None = None


def init_experiment_run(
    metadata: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> ExperimentRun:
    """
    Create experiments/<timestamp>/{results,plots} and write manifest.json.

    Subsequent CSV/plot writes use the new directories for this process.
    """
    global RESULTS_DIR, PLOTS_DIR, _active_run

    run_id = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = EXPERIMENTS_ROOT / run_id
    results_dir = root / "results"
    plots_dir = root / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **(metadata or {}),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run = ExperimentRun(run_id=run_id, root=root, results_dir=results_dir, plots_dir=plots_dir)
    RESULTS_DIR = results_dir
    PLOTS_DIR = plots_dir
    _active_run = run
    return run


def use_experiment_dir(path: Path) -> ExperimentRun:
    """Point RESULTS_DIR / PLOTS_DIR at an existing experiment (for regenerate_plots)."""
    global RESULTS_DIR, PLOTS_DIR, _active_run

    path = path.resolve()
    if path.name == "results":
        root = path.parent
        results_dir = path
        plots_dir = root / "plots"
    elif (path / "results").is_dir():
        root = path
        results_dir = root / "results"
        plots_dir = root / "plots"
    else:
        raise FileNotFoundError(f"No results/ directory under {path}")

    plots_dir.mkdir(parents=True, exist_ok=True)
    run_id = root.name
    run = ExperimentRun(run_id=run_id, root=root, results_dir=results_dir, plots_dir=plots_dir)
    RESULTS_DIR = results_dir
    PLOTS_DIR = plots_dir
    _active_run = run
    return run


def active_experiment() -> ExperimentRun | None:
    return _active_run


@dataclass(frozen=True)
class Dataset:
    """Deterministic synthetic vector corpus and query set."""

    name: str
    dimension: int
    vectors: np.ndarray  # shape (N, D), float32
    queries: np.ndarray  # shape (Q, D), float32
    ids: np.ndarray  # shape (N,), int64

    @property
    def num_vectors(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def num_queries(self) -> int:
        return int(self.queries.shape[0])


def generate_dataset(
    dimension: int,
    num_vectors: int = NUM_VECTORS,
    num_queries: int = NUM_QUERIES,
    seed: int = RANDOM_SEED,
) -> Dataset:
    """
    Create a reproducible synthetic dataset.

    Vectors are drawn from a standard normal distribution and L2-normalized
    so that dot-product scores remain well-conditioned for high D.
    """
    rng = np.random.default_rng(seed + dimension)
    vectors = rng.standard_normal((num_vectors, dimension), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8

    queries = rng.standard_normal((num_queries, dimension), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8

    ids = np.arange(num_vectors, dtype=np.int64)
    return Dataset(
        name=f"synth_d{dimension}_n{num_vectors}",
        dimension=dimension,
        vectors=vectors,
        queries=queries,
        ids=ids,
    )


# ---------------------------------------------------------------------------
# Score decomposition (mirrors XQdrant Dot distance semantics)
# ---------------------------------------------------------------------------


def dot_contributions(query: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Per-dimension dot-product terms q_i * v_i."""
    return query * vector


def top_m_dims_full_sort(contributions: np.ndarray, m: int) -> np.ndarray:
    """
    Client-side post-query extraction baseline: full sort O(D log D).

    Returns indices of the top-m dimensions by absolute contribution.
    """
    m = min(m, contributions.size)
    order = np.argsort(np.abs(contributions))[::-1]
    return order[:m]


def top_m_dims_bounded_heap(contributions: np.ndarray, m: int) -> np.ndarray:
    """
    XQdrant-style bounded max-heap selection: O(D log m).

    Uses Python's heapq on the smallest absolute contributions to maintain
    a size-m reservoir of the largest contributors.
    """
    m = min(m, contributions.size)
    if m == 0:
        return np.array([], dtype=np.int64)

    heap: list[tuple[float, int]] = []
    for idx, value in enumerate(contributions):
        key = (abs(float(value)), int(idx))
        if len(heap) < m:
            heapq.heappush(heap, key)
        elif key > heap[0]:
            heapq.heapreplace(heap, key)

    ranked = sorted(heap, key=lambda item: item[0], reverse=True)
    return np.array([idx for _, idx in ranked], dtype=np.int64)


# ---------------------------------------------------------------------------
# Ground truth and recall
# ---------------------------------------------------------------------------


def brute_force_top_k(
    query: np.ndarray,
    vectors: np.ndarray,
    k: int,
    dim_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact top-k by dot product.

    When dim_indices is provided, only those dimensions contribute to the score
    (matches XQdrant masked / focus subspace semantics).
    """
    if dim_indices is None:
        scores = vectors @ query
    else:
        sub_q = query[dim_indices]
        sub_v = vectors[:, dim_indices]
        scores = sub_v @ sub_q
    k = min(k, scores.shape[0])
    if k == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return top_idx.astype(np.int64), scores[top_idx].astype(np.float32)


def recall_at_k(retrieved_ids: Sequence[int], ground_truth_ids: Sequence[int], k: int) -> float:
    """Standard Recall@K = |retrieved ∩ ground_truth| / K."""
    if k <= 0:
        return 1.0
    retrieved = set(int(x) for x in retrieved_ids[:k])
    truth = set(int(x) for x in ground_truth_ids[:k])
    if not truth:
        return 1.0
    return len(retrieved & truth) / float(k)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


@dataclass
class LatencyStats:
    samples_ns: list[int] = field(default_factory=list)

    def record(self, elapsed_ns: int) -> None:
        self.samples_ns.append(elapsed_ns)

    def extend(self, other: "LatencyStats") -> None:
        self.samples_ns.extend(other.samples_ns)

    def percentiles_ms(self) -> dict[str, float]:
        if not self.samples_ns:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        arr = np.asarray(self.samples_ns, dtype=np.float64)
        return {
            "p50": float(np.percentile(arr, 50) / 1e6),
            "p95": float(np.percentile(arr, 95) / 1e6),
            "p99": float(np.percentile(arr, 99) / 1e6),
            "mean": float(arr.mean() / 1e6),
        }


def timed_call(func, *args, **kwargs) -> tuple[object, int]:
    """Execute func and return (result, elapsed_nanoseconds)."""
    start = time.perf_counter_ns()
    result = func(*args, **kwargs)
    end = time.perf_counter_ns()
    return result, end - start


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    ensure_output_dirs()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def detect_physical_cores() -> int:
    """Best-effort physical core count for throughput scaling."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError):
        pass

    if sys.platform == "darwin":
        try:
            import subprocess

            out = subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"],
                text=True,
            )
            return max(1, int(out.strip()))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    count = os.cpu_count() or 1
    return max(1, count // 2)
