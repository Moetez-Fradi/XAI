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
# Extra (non-plain) hot paths — attribution, focus rescore, focus masked — are
# warmed for this many queries so their page cache / scratch buffers are hot
# before timing. Kept smaller than WARMUP_QUERIES since these are only needed to
# defeat first-touch artifacts, not to converge latency.
WARMUP_EXTRA = 200

DIMENSIONS = (768, 1536)
EF_SEARCH_VALUES = [16, 22, 32, 45, 64, 90, 128, 181, 256, 362, 512]
THREAD_COUNTS = [1, 2, 4, 8, 16]
ATTRIBUTION_DEPTHS = [1, 5, 10, 20, 50]
SUBSPACE_RATIOS = [0.1, 0.25, 0.5, 0.75]

DEFAULT_DISTANCE = "Dot"
DEFAULT_EF_SEARCH = 128
DEFAULT_ATTRIBUTION_M = 10

# HNSW index parameters used when provisioning collections (recorded in the run
# manifest for reproducibility).
HNSW_M = 16
HNSW_EF_CONSTRUCT = 200

PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = PACKAGE_DIR / "experiments"
# Default location for the SIFT1M corpus (sift_base.fvecs, sift_query.fvecs,
# sift_groundtruth.ivecs). Override with --sift-dir / SIFT_DIR.
SIFT_DIR = Path(os.environ.get("SIFT_DIR", str(PACKAGE_DIR / "data" / "sift")))

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
    """A vector corpus + query set with its native distance metric."""

    name: str
    dimension: int
    vectors: np.ndarray  # shape (N, D), float32
    queries: np.ndarray  # shape (Q, D), float32
    ids: np.ndarray  # shape (N,), int64
    distance: str = "Dot"  # Qdrant metric: "Dot" | "Cosine" | "Euclid" | "Manhattan"
    ground_truth: np.ndarray | None = None  # optional (Q, G) full-space top-G ids

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
        distance=DEFAULT_DISTANCE,
    )


# ---------------------------------------------------------------------------
# SIFT1M loader (.fvecs / .ivecs, Euclidean)
# ---------------------------------------------------------------------------


def read_fvecs(path: Path, limit: int | None = None) -> np.ndarray:
    """
    Read a .fvecs file (TEXMEX format): each record is int32 dim d followed by d float32.

    Returns a (num, d) float32 array. ``limit`` caps the number of records read.
    """
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"Empty or missing .fvecs file: {path}")
    dim = int(raw[0])
    record = dim + 1  # 1 int32 header + dim values
    num = raw.size // record
    data = raw.reshape(num, record)
    if not np.all(data[:, 0] == dim):
        raise ValueError(f"Inconsistent vector dimension in {path}")
    vecs = data[:, 1:].view(np.float32)
    if limit is not None:
        vecs = vecs[:limit]
    return np.ascontiguousarray(vecs, dtype=np.float32)


def read_ivecs(path: Path, limit: int | None = None) -> np.ndarray:
    """Read an .ivecs file (int32 records); returns (num, d) int64 array."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"Empty or missing .ivecs file: {path}")
    dim = int(raw[0])
    record = dim + 1
    num = raw.size // record
    data = raw.reshape(num, record)[:, 1:]
    if limit is not None:
        data = data[:limit]
    return np.ascontiguousarray(data, dtype=np.int64)


def load_sift1m(
    data_dir: Path | str = SIFT_DIR,
    num_vectors: int | None = None,
    num_queries: int | None = None,
) -> Dataset:
    """
    Load the SIFT1M corpus (Euclidean, D=128).

    Expects ``sift_base.fvecs``, ``sift_query.fvecs`` and ``sift_groundtruth.ivecs`` under
    ``data_dir`` (see research/../fetch_sift.sh). The bundled ground truth is full-corpus,
    full-query top-100 by L2; it is only attached when the corpus is NOT subsampled, otherwise
    ground truth is recomputed on demand.
    """
    data_dir = Path(data_dir)
    base_path = data_dir / "sift_base.fvecs"
    query_path = data_dir / "sift_query.fvecs"
    gt_path = data_dir / "sift_groundtruth.ivecs"
    if not base_path.exists():
        raise FileNotFoundError(
            f"SIFT base vectors not found at {base_path}. "
            f"Download with DB_systems/fetch_sift.sh (see README)."
        )

    vectors = read_fvecs(base_path, limit=num_vectors)
    queries = read_fvecs(query_path, limit=num_queries)
    dimension = int(vectors.shape[1])

    # Bundled GT indexes into the full base, so it is only valid when the base is NOT
    # subsampled. Subsampling queries is fine — GT rows align with query rows, so slice them.
    ground_truth = None
    if num_vectors is None and gt_path.exists():
        ground_truth = read_ivecs(gt_path)
        if num_queries is not None:
            ground_truth = ground_truth[:num_queries]

    ids = np.arange(vectors.shape[0], dtype=np.int64)
    return Dataset(
        name=f"sift1m_n{vectors.shape[0]}",
        dimension=dimension,
        vectors=vectors,
        queries=queries,
        ids=ids,
        distance="Euclid",
        ground_truth=ground_truth,
    )


def build_dataset(
    kind: str = "synthetic",
    dimension: int = 768,
    *,
    sift_dir: Path | str = SIFT_DIR,
    num_vectors: int | None = None,
    num_queries: int | None = None,
    seed: int = RANDOM_SEED,
) -> Dataset:
    """Factory: 'synthetic' (Dot, chosen D) or 'sift1m' (Euclid, D=128)."""
    kind = kind.lower()
    if kind in ("synthetic", "synth", "random"):
        return generate_dataset(
            dimension=dimension,
            num_vectors=num_vectors or NUM_VECTORS,
            num_queries=num_queries or NUM_QUERIES,
            seed=seed,
        )
    if kind in ("sift", "sift1m"):
        return load_sift1m(sift_dir, num_vectors=num_vectors, num_queries=num_queries)
    raise ValueError(f"Unknown dataset kind: {kind!r}. Use 'synthetic' or 'sift1m'.")


# ---------------------------------------------------------------------------
# Score decomposition (mirrors XQdrant Dot distance semantics)
# ---------------------------------------------------------------------------


def dot_contributions(query: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Per-dimension dot-product terms q_i * v_i."""
    return query * vector


def score_contributions(
    query: np.ndarray, vector: np.ndarray, distance: str = "Dot"
) -> np.ndarray:
    """
    Per-dimension score-decomposition terms, matching XQdrant's dims_explained math.

    | Dot       | q_i * v_i        |
    | Cosine    | q̂_i * v̂_i       (L2-normalized) |
    | Euclid    | (q_i - v_i)^2    |
    | Manhattan | |q_i - v_i|      |
    """
    d = distance.lower()
    if d == "dot":
        return query * vector
    if d == "cosine":
        q = query / (np.linalg.norm(query) + 1e-8)
        v = vector / (np.linalg.norm(vector) + 1e-8)
        return q * v
    if d in ("euclid", "euclidean", "l2"):
        diff = query - vector
        return diff * diff
    if d == "manhattan":
        return np.abs(query - vector)
    raise ValueError(f"Unknown distance for contributions: {distance!r}")


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
    distance: str = "Dot",
    vectors_norm_sq: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact top-k under the given distance metric.

    - Dot / Cosine: higher score = better (nearest = largest similarity).
    - Euclid: smaller squared L2 = better (nearest = smallest distance).

    When ``dim_indices`` is provided, only those dimensions contribute (matches XQdrant
    masked / focus subspace semantics). For repeated calls over the same subspace at scale,
    prefer ``SubspaceGT`` which precomputes the projection and norms once.

    ``vectors_norm_sq`` optionally supplies precomputed row norms (for Euclid) matching the
    dimensions actually used (full or subspace).
    """
    if dim_indices is not None:
        sub_q = query[dim_indices]
        sub_v = vectors[:, dim_indices]
    else:
        sub_q = query
        sub_v = vectors

    n = sub_v.shape[0]
    k = min(k, n)
    if k == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    d = distance.lower()
    if d in ("dot",):
        scores = sub_v @ sub_q
        largest = True
    elif d == "cosine":
        vn = sub_v / (np.linalg.norm(sub_v, axis=1, keepdims=True) + 1e-8)
        qn = sub_q / (np.linalg.norm(sub_q) + 1e-8)
        scores = vn @ qn
        largest = True
    elif d in ("euclid", "euclidean", "l2"):
        norm_sq = vectors_norm_sq if vectors_norm_sq is not None else np.einsum("ij,ij->i", sub_v, sub_v)
        # ||v - q||^2 = ||v||^2 - 2 v·q + ||q||^2; the +||q||^2 term is constant per query.
        scores = norm_sq - 2.0 * (sub_v @ sub_q)
        largest = False
    else:
        raise ValueError(f"Unsupported distance for brute force: {distance!r}")

    if largest:
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    else:
        top_idx = np.argpartition(scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(scores[top_idx])]
    return top_idx.astype(np.int64), scores[top_idx].astype(np.float32)


class SubspaceGT:
    """
    Precomputed exact-nearest-neighbour helper over a fixed subspace.

    Projecting 1M vectors onto the focus dims once (instead of per query) is the difference
    between a fast and an unusable ground-truth loop at SIFT1M scale.
    """

    def __init__(self, vectors: np.ndarray, dim_indices: np.ndarray, distance: str = "Dot"):
        self.dims = np.asarray(dim_indices)
        self.distance = distance
        self.sub_v = np.ascontiguousarray(vectors[:, self.dims])
        if distance.lower() in ("euclid", "euclidean", "l2"):
            self.norm_sq = np.einsum("ij,ij->i", self.sub_v, self.sub_v)
        else:
            self.norm_sq = None

    def top_k(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sub_q = query[self.dims]
        return brute_force_top_k(
            sub_q, self.sub_v, k, distance=self.distance, vectors_norm_sq=self.norm_sq
        )


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


def git_commit() -> str | None:
    """Best-effort short+long git SHA of the working tree (for the manifest)."""
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PACKAGE_DIR),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


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
