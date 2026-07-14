"""
Shared helpers for the XQdrant masked-traversal *research* scripts.

This is the reusable core for everything under ``DB_systems/research/``. The
per-option run scripts (option1/, option2/, ...) are intentionally thin and
import from here so that:

  * every experiment writes a **self-describing, timestamped** result folder
    (``experiments/<ts>__<step_id>__<metric>/``) with a manifest recording the
    step, metric, backend mode, URLs, sweep parameters, and whether the run
    depended on an XQdrant Rust change that may not be merged yet;
  * request bodies for the *proposed* focus fields (``mask_from_layer``,
    ``alpha``, ``verify``) live in one place — update them here if the Rust API
    names change;
  * a config that hits an unimplemented XQdrant field is recorded as
    ``status="unsupported"`` instead of crashing the whole sweep.

It reuses the existing suite library (``bench_common`` / ``bench_backends``) for
dataset generation, provisioning, ground truth, recall, and CSV writing.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

# --- make the shared suite library importable from any research subfolder ----
DB_SYSTEMS_DIR = Path(__file__).resolve().parents[1]
if str(DB_SYSTEMS_DIR) not in sys.path:
    sys.path.insert(0, str(DB_SYSTEMS_DIR))

import bench_common  # noqa: E402
from bench_common import (  # noqa: E402
    SIFT_DIR,
    Dataset,
    LatencyStats,
    SubspaceGT,
    brute_force_top_k,
    build_dataset,
    generate_dataset,
    init_experiment_run,
    recall_at_k,
    write_csv,
)
from bench_backends import HttpBackend, SearchResult, make_backend  # noqa: E402

try:
    import requests  # noqa: E402
except ImportError:  # pragma: no cover - requests only needed for http mode
    requests = None  # type: ignore


# ---------------------------------------------------------------------------
# Self-describing, timestamped run directories
# ---------------------------------------------------------------------------


def init_research_run(
    step_id: str,
    metric: str,
    *,
    mode: str,
    dimensions: Sequence[int],
    queries: int,
    qdrant_url: str | None = None,
    xqdrant_url: str | None = None,
    requires_xqdrant_change: str | None = None,
    sweep: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    run_id: str | None = None,
):
    """
    Create ``experiments/<ts>__<step_id>__<metric>/`` and return the ExperimentRun.

    The folder name embeds *step* and *metric* so that, when summarizing many
    runs later, you never have to guess which experiment produced which CSV.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_id is None:
        run_id = f"{timestamp}__{step_id}__{metric}"
    metadata: dict[str, Any] = {
        "step_id": step_id,
        "metric": metric,
        "mode": mode,
        "dimensions": list(dimensions),
        "queries": queries,
        "qdrant_url": qdrant_url,
        "xqdrant_url": xqdrant_url,
        "requires_xqdrant_change": requires_xqdrant_change or "none",
        "sweep": sweep or {},
    }
    if extra:
        metadata.update(extra)
    return init_experiment_run(run_id=run_id, metadata=metadata)


def subspace_indices(dimension: int, ratio: float, seed: int = 0) -> np.ndarray:
    """Deterministic sorted focus-dimension subset (matches bench_tests convention)."""
    rng = np.random.default_rng(seed + int(ratio * 10_000))
    count = max(1, int(round(dimension * ratio)))
    return np.sort(rng.choice(dimension, size=count, replace=False))


# ---------------------------------------------------------------------------
# Dataset selection shared across research scripts
# ---------------------------------------------------------------------------


def add_dataset_args(parser) -> None:
    """Add --dataset / --sift-dir / --num-vectors to a research script's argparser."""
    parser.add_argument("--dataset", choices=["synthetic", "sift1m"], default="synthetic",
                        help="Corpus: synthetic (Dot) or sift1m (Euclid, D=128)")
    parser.add_argument("--sift-dir", default=str(SIFT_DIR),
                        help="Directory with sift_base.fvecs / sift_query.fvecs / sift_groundtruth.ivecs")
    parser.add_argument("--num-vectors", type=int, default=None,
                        help="Cap corpus size (subsample SIFT1M for quick runs)")


def iter_datasets(args):
    """
    Yield the dataset(s) for a run. SIFT1M is one fixed corpus (D=128); synthetic sweeps
    ``args.dimensions``. Works whether or not add_dataset_args was used (defaults to synthetic).
    """
    kind = getattr(args, "dataset", "synthetic")
    num_vectors = getattr(args, "num_vectors", None)
    num_queries = getattr(args, "queries", None)
    if kind == "sift1m":
        yield build_dataset("sift1m", sift_dir=getattr(args, "sift_dir", str(SIFT_DIR)),
                            num_vectors=num_vectors, num_queries=num_queries)
    else:
        for dim in args.dimensions:
            yield build_dataset(
                "synthetic",
                dimension=dim,
                num_vectors=num_vectors,
                num_queries=num_queries,
            )


def full_norms_sq(dataset: Dataset) -> np.ndarray | None:
    """Precompute row norms for Euclid full-space GT (None for similarity metrics)."""
    if dataset.distance.lower() in ("euclid", "euclidean", "l2"):
        return np.einsum("ij,ij->i", dataset.vectors, dataset.vectors)
    return None


def full_top_k(dataset: Dataset, idx: int, query: np.ndarray, k: int,
               norms_sq: np.ndarray | None = None) -> np.ndarray:
    """Full-space top-k ids: bundled ground truth if present, else exact under the metric."""
    if dataset.ground_truth is not None and idx < len(dataset.ground_truth):
        return dataset.ground_truth[idx][:k]
    ids, _ = brute_force_top_k(query, dataset.vectors, k,
                               distance=dataset.distance, vectors_norm_sq=norms_sq)
    return ids


# ---------------------------------------------------------------------------
# Focus request-body builders (proposed REST field names)
# ---------------------------------------------------------------------------
#
# These names MUST match the XQdrant Rust change for each option. If you rename a
# field on the Rust side, change it here once and every research script follows.


def focus_body(
    query: np.ndarray,
    k: int,
    ef_search: int,
    dims: np.ndarray | None = None,
    *,
    masked: bool = False,
    mask_from_layer: int | None = None,
    alpha: float | None = None,
    verify: bool | None = None,
    candidates_limit: int | None = None,
) -> dict[str, Any]:
    """Build a ``/points/query`` body, optionally with focus + proposed knobs."""
    query_obj: dict[str, Any] = {"nearest": query.tolist()}
    if dims is not None:
        focus: dict[str, Any] = {"dims": [int(d) for d in np.asarray(dims).tolist()]}
        if masked:
            focus["masked"] = True
        if mask_from_layer is not None:
            focus["mask_from_layer"] = int(mask_from_layer)  # Option 2
        if alpha is not None:
            focus["alpha"] = float(alpha)  # Option 4
        if verify is not None:
            focus["verify"] = bool(verify)  # X3 verify pass
        if candidates_limit is not None:
            focus["candidates_limit"] = int(candidates_limit)
        query_obj["focus"] = focus
    return {
        "query": query_obj,
        "limit": k,
        "params": {"hnsw_ef": ef_search},
    }


# ---------------------------------------------------------------------------
# HTTP client for arbitrary bodies (reuses HttpBackend for provisioning)
# ---------------------------------------------------------------------------


@dataclass
class QueryOutcome:
    """Result of a single research query, with graceful error capture."""

    result: SearchResult | None
    latency_ns: int
    status: str = "ok"  # "ok" | "unsupported" | "error"
    detail: str = ""


class ResearchClient:
    """
    Thin wrapper over the suite's HttpBackend that can POST *custom* query
    bodies (needed for the proposed focus knobs) and captures unsupported-field
    errors instead of aborting a sweep.
    """

    def __init__(
        self,
        dataset: Dataset,
        qdrant_url: str,
        xqdrant_url: str | None = None,
        provision: bool = True,
    ):
        self.dataset = dataset
        self._backend = HttpBackend(
            dataset,
            qdrant_url=qdrant_url,
            xqdrant_url=xqdrant_url,
            setup_collection=provision,
        )

    @property
    def collection(self) -> str:
        return self._backend.collection_name

    @property
    def qdrant_url(self) -> str:
        return self._backend.qdrant_url

    @property
    def xqdrant_url(self) -> str:
        return self._backend.xqdrant_url

    def warmup(self, queries: np.ndarray, count: int) -> None:
        self._backend.warmup(queries, count)

    def wait_for_indexing(self, expected_points: int | None = None) -> None:
        """Block until the collection is green / fully indexed (HTTP only)."""
        self._backend.wait_for_indexing(expected_points=expected_points)

    def query(self, body: dict[str, Any], base_url: str | None = None) -> QueryOutcome:
        base = base_url or self._backend.xqdrant_url
        path = f"/collections/{self.collection}/points/query"
        start = time.perf_counter_ns()
        try:
            payload = self._backend._request(base, "POST", path, body)
        except Exception as exc:  # noqa: BLE001 - want to classify below
            elapsed = time.perf_counter_ns() - start
            detail = str(exc)
            status = "unsupported" if _looks_unsupported(detail) else "error"
            return QueryOutcome(None, elapsed, status=status, detail=detail[:300])
        elapsed = time.perf_counter_ns() - start
        return QueryOutcome(self._backend._parse_points(payload), elapsed)


def _looks_unsupported(detail: str) -> bool:
    needles = (
        "unknown field",
        "unknown variant",
        "unimplemented",
        "not supported",
        "unsupported",
        "invalid type",
        "missing field",
        "400",
        "422",
    )
    low = detail.lower()
    return any(n in low for n in needles)


# ---------------------------------------------------------------------------
# Simulated model (pipeline validation before the Rust change lands)
# ---------------------------------------------------------------------------
#
# Clearly-tagged, monotonic cost/recall models so the sweep + plot pipeline can
# be validated offline. NOT a substitute for live HNSW measurement — every CSV
# row it produces carries simulated=1 and plots are titled "[SIMULATED]".


def simulated_masked(
    query: np.ndarray,
    dataset: Dataset,
    k: int,
    dims: np.ndarray,
    *,
    mask_from_layer: int | None = None,
    max_layer: int = 4,
    alpha: float | None = None,
) -> tuple[SearchResult, int, float]:
    """
    Return (result, latency_ns, recall) approximating masked traversal.

    Recall degrades as the subspace shrinks and as more layers are masked;
    latency drops with the masked fraction. Used only for --mode simulated.
    """
    dimension = dataset.dimension
    ratio = len(dims) / max(1, dimension)

    # Exact subspace top-k as the "returned" set, then perturb to model misrouting.
    gt_ids, _ = brute_force_top_k(query, dataset.vectors, k, dim_indices=dims,
                                  distance=dataset.distance)

    # Fraction of layers actually masked. Option 2 semantics: layer L is masked iff
    # L < mask_from_layer. So mask_from_layer=0 masks none (best recall); a large
    # cutoff masks every layer (Option 1 / worst recall). None → fully masked.
    if mask_from_layer is None:
        masked_frac = 1.0
    else:
        masked_frac = max(0.0, min(1.0, mask_from_layer / max(1, max_layer)))

    # Navigability penalty grows as we mask more layers and shrink the subspace.
    penalty = masked_frac * (1.0 - ratio)
    if alpha is not None:
        penalty *= (1.0 - alpha)  # blending toward full distance recovers recall
    recall = float(np.clip(1.0 - 0.9 * penalty, 0.0, 1.0))

    # Model returned ids: keep a recall-fraction of gt, fill rest with near-misses.
    keep = int(round(recall * k))
    rng = np.random.default_rng(int(abs(query.sum()) * 1e6) % (2**32))
    filler = rng.choice(dataset.num_vectors, size=max(0, k - keep), replace=False)
    ids = list(gt_ids[:keep].tolist()) + [int(x) for x in filler]

    base_ns = 1_000_000  # 1ms nominal full-vector comparison budget
    work = masked_frac * ratio + (1.0 - masked_frac)  # masked part cheap, full part full
    latency_ns = int(base_ns * max(0.2, work))
    return SearchResult(ids=ids, scores=[0.0] * len(ids)), latency_ns, recall


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot", "classic"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    return plt


def save_fig(fig, stem: str) -> Path:
    """Save a figure as PNG + PDF into the active experiment's plots dir."""
    bench_common.ensure_output_dirs()
    png = bench_common.PLOTS_DIR / f"{stem}.png"
    pdf = bench_common.PLOTS_DIR / f"{stem}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return png


def line_plot(
    x: Sequence[float],
    series: dict[str, Sequence[float]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    stem: str,
    hline: float | None = None,
) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, ys in series.items():
        ax.plot(x, ys, marker="o", linewidth=2, label=label)
    if hline is not None:
        ax.axhline(hline, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save_fig(fig, stem)


def heatmap(
    matrix: np.ndarray,
    *,
    row_labels: Sequence[Any],
    col_labels: Sequence[Any],
    xlabel: str,
    ylabel: str,
    title: str,
    stem: str,
    cbar_label: str,
    fmt: str = "{:.2f}",
) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(1.4 * len(col_labels) + 3, 1.0 * len(row_labels) + 3))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        color="white" if val < np.nanmax(matrix) * 0.6 else "black",
                        fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return save_fig(fig, stem)


__all__ = [
    "Dataset",
    "LatencyStats",
    "QueryOutcome",
    "ResearchClient",
    "SubspaceGT",
    "add_dataset_args",
    "bench_common",
    "brute_force_top_k",
    "build_dataset",
    "focus_body",
    "full_norms_sq",
    "full_top_k",
    "generate_dataset",
    "heatmap",
    "init_research_run",
    "iter_datasets",
    "line_plot",
    "make_backend",
    "recall_at_k",
    "save_fig",
    "simulated_masked",
    "subspace_indices",
    "write_csv",
]
