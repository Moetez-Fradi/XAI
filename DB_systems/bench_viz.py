"""
Publication-quality chart generation for XQdrant benchmark results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bench_common
from bench_common import ensure_output_dirs


STYLE = "seaborn-v0_8-whitegrid"
MARKERS = {
    "vanilla": "o",
    "xqdrant": "s",
    "post_query": "^",
    "speedup": "x",
    "rescore": "D",
    "masked": "P",
}
COLORS = {
    "vanilla": "#1f77b4",
    "xqdrant": "#d62728",
    "post_query": "#2ca02c",
    "speedup": "#9467bd",
    "rescore": "#ff7f0e",
    "masked": "#17becf",
}


def _apply_pub_style() -> None:
    for style in (STYLE, "seaborn-whitegrid", "ggplot", "classic"):
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
            "legend.fontsize": 12,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    import csv

    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _col(rows: list[dict], name: str, default: float = 0.0) -> list[float]:
    out: list[float] = []
    for r in rows:
        if name in r and r[name] not in ("", None):
            out.append(float(r[name]))
        else:
            out.append(default)
    return out


def _errorbar(ax, x, y, yerr, *, marker, color, label, linewidth=2):
    """Plot mean with std error bars when yerr is non-trivial; else a plain line."""
    yerr_arr = np.asarray(yerr, dtype=np.float64)
    if np.all(~np.isfinite(yerr_arr)) or np.nanmax(yerr_arr) <= 0:
        ax.plot(x, y, marker=marker, color=color, linewidth=linewidth, label=label)
        return
    ax.errorbar(
        x,
        y,
        yerr=yerr_arr,
        marker=marker,
        color=color,
        linewidth=linewidth,
        capsize=3,
        elinewidth=1.2,
        label=label,
    )


def plot_latency_recall(dimension: int, output_stem: str = "plot1_latency_recall") -> Path | None:
    """
    Plot 1: X-axis = 1 - Recall@K, Y-axis = Latency (ms).
    Lines for Vanilla vs XQdrant (p50 mean ± std across trials).
    """
    rows = _read_csv(bench_common.RESULTS_DIR / f"latency_recall_d{dimension}.csv")
    if not rows:
        return None

    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    one_minus_vanilla = [1.0 - float(r["vanilla_recall_at_k"]) for r in rows]
    one_minus_xq = [1.0 - float(r["xqdrant_recall_at_k"]) for r in rows]
    vanilla_lat = _col(rows, "vanilla_p50_ms")
    xq_lat = _col(rows, "xqdrant_p50_ms")
    vanilla_err = _col(rows, "vanilla_p50_ms_std")
    xq_err = _col(rows, "xqdrant_p50_ms_std")

    _errorbar(
        ax,
        one_minus_vanilla,
        vanilla_lat,
        vanilla_err,
        marker=MARKERS["vanilla"],
        color=COLORS["vanilla"],
        label="Vanilla Qdrant",
    )
    _errorbar(
        ax,
        one_minus_xq,
        xq_lat,
        xq_err,
        marker=MARKERS["xqdrant"],
        color=COLORS["xqdrant"],
        label="XQdrant (in-database)",
    )

    ax.set_xlabel("1 − Recall@K")
    ax.set_ylabel("Latency (ms, mean ± std)")
    ax.set_title(f"Latency vs. Recall Trade-off (D={dimension})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ensure_output_dirs()
    png_path = bench_common.PLOTS_DIR / f"{output_stem}_d{dimension}.png"
    pdf_path = bench_common.PLOTS_DIR / f"{output_stem}_d{dimension}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def plot_throughput(dimension: int, output_stem: str = "plot2_throughput") -> Path | None:
    """
    Plot 2: X-axis = Concurrent Threads, Y-axis = Throughput (QPS).
    """
    rows = _read_csv(bench_common.RESULTS_DIR / f"throughput_d{dimension}.csv")
    if not rows:
        return None

    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    threads = [int(r["threads"]) for r in rows]
    qps = _col(rows, "qps")
    qps_err = _col(rows, "qps_std")

    _errorbar(
        ax,
        threads,
        qps,
        qps_err,
        marker=MARKERS["vanilla"],
        color=COLORS["vanilla"],
        label="Vanilla Qdrant",
    )

    ax.set_xlabel("Concurrent Threads")
    ax.set_ylabel("Throughput (QPS, mean ± std)")
    ax.set_title(f"Multi-threaded Query Scalability (D={dimension})")
    ax.set_xticks(threads)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ensure_output_dirs()
    png_path = bench_common.PLOTS_DIR / f"{output_stem}_d{dimension}.png"
    pdf_path = bench_common.PLOTS_DIR / f"{output_stem}_d{dimension}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def plot_attribution_depth(dimension: int, output_stem: str = "plot3_attribution_depth") -> Path | None:
    """
    Plot 3: X-axis = Attribution Depth (m), Y-axis = Latency (ms).
    Compares XQdrant vs Post-Query extraction.
    """
    rows = _read_csv(bench_common.RESULTS_DIR / f"attribution_depth_d{dimension}.csv")
    if not rows:
        return None

    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    m_vals = [int(r["m"]) for r in rows]
    post_lat = _col(rows, "post_query_p50_ms")
    xq_lat = _col(rows, "xqdrant_p50_ms")
    post_err = _col(rows, "post_query_p50_ms_std")
    xq_err = _col(rows, "xqdrant_p50_ms_std")
    speedups = _col(rows, "speedup_p50")
    mean_speedup = float(np.mean(speedups))

    # Shaded band between the two lines highlights the latency saved in-database.
    ax.fill_between(
        m_vals,
        xq_lat,
        post_lat,
        alpha=0.18,
        color=COLORS["post_query"],
        linewidth=0,
    )

    _errorbar(
        ax,
        m_vals,
        post_lat,
        post_err,
        marker=MARKERS["post_query"],
        color=COLORS["post_query"],
        label="Post-Query Extraction",
        linewidth=2.2,
    )
    _errorbar(
        ax,
        m_vals,
        xq_lat,
        xq_err,
        marker=MARKERS["xqdrant"],
        color=COLORS["xqdrant"],
        label="XQdrant (in-database)",
        linewidth=2.2,
    )

    y_min = max(0.0, min(xq_lat) * 0.85)
    y_max = max(post_lat) * 1.08
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Attribution Depth (m)")
    ax.set_ylabel("p50 Latency (ms, mean ± std)")
    ax.set_title(f"Attribution Depth Scaling (D={dimension})")
    ax.set_xticks(m_vals)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        framealpha=0.95,
    )

    ax.text(
        0.03,
        0.97,
        f"Mean speedup: {mean_speedup:.2f}×",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    fig.tight_layout()

    ensure_output_dirs()
    png_path = bench_common.PLOTS_DIR / f"{output_stem}_d{dimension}.png"
    pdf_path = bench_common.PLOTS_DIR / f"{output_stem}_d{dimension}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def _read_subspace_rescore_csv(dimension: int) -> list[dict]:
    for name in (f"subspace_rescore_d{dimension}.csv", f"subspace_pruning_d{dimension}.csv"):
        rows = _read_csv(bench_common.RESULTS_DIR / name)
        if rows:
            return rows
    return []


def plot_subspace_speedup(dimension: int, output_stem: str = "plot4_subspace_rescore") -> Path | None:
    """
    Plot 4: Focus *rescore* latency vs full-dimension baseline.
    """
    rows = _read_subspace_rescore_csv(dimension)
    if not rows:
        return None

    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    ratios = [float(r["subspace_ratio"]) for r in rows]
    speedups = _col(rows, "speedup_factor")
    speed_err = _col(rows, "speedup_factor_std")

    _errorbar(
        ax,
        ratios,
        speedups,
        speed_err,
        marker=MARKERS["rescore"],
        color=COLORS["rescore"],
        label="Focus Rescore (preselect + rescore)",
    )

    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="Full-dimension baseline")
    ax.set_xlabel("Subspace Ratio (D_sub / D)")
    ax.set_ylabel("Latency Speedup Factor (mean ± std)")
    ax.set_title(f"Focus Rescore Latency (D={dimension})")
    ax.set_xticks(ratios)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ensure_output_dirs()
    png_path = bench_common.PLOTS_DIR / f"{output_stem}_d{dimension}.png"
    pdf_path = bench_common.PLOTS_DIR / f"{output_stem}_d{dimension}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def plot_masked_subspace(dimension: int, output_stem: str = "plot5_masked_subspace") -> Path | None:
    """
    Plot 5: Masked HNSW vs rescore — speedup (top) and subspace recall (bottom).
    """
    rows = _read_csv(bench_common.RESULTS_DIR / f"masked_subspace_d{dimension}.csv")
    if not rows:
        return None

    _apply_pub_style()
    fig, (ax_speed, ax_recall) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ratios = [float(r["subspace_ratio"]) for r in rows]
    masked_speedup = _col(rows, "masked_speedup")
    rescore_speedup = _col(rows, "rescore_speedup")
    masked_recall = _col(rows, "masked_recall_at_k")
    rescore_recall = _col(rows, "rescore_recall_at_k")
    masked_speed_err = _col(rows, "masked_speedup_std")
    rescore_speed_err = _col(rows, "rescore_speedup_std")
    masked_rec_err = _col(rows, "masked_recall_at_k_std")
    rescore_rec_err = _col(rows, "rescore_recall_at_k_std")

    _errorbar(
        ax_speed,
        ratios,
        masked_speedup,
        masked_speed_err,
        marker=MARKERS["masked"],
        color=COLORS["masked"],
        label="Masked HNSW (focus.masked)",
        linewidth=2.2,
    )
    _errorbar(
        ax_speed,
        ratios,
        rescore_speedup,
        rescore_speed_err,
        marker=MARKERS["rescore"],
        color=COLORS["rescore"],
        label="Focus Rescore",
        linewidth=2.2,
    )
    ax_speed.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax_speed.set_ylabel("Latency Speedup vs Full (mean ± std)")
    ax_speed.set_title(f"Masked Subspace Search (D={dimension})")
    ax_speed.legend(loc="best")
    ax_speed.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    _errorbar(
        ax_recall,
        ratios,
        masked_recall,
        masked_rec_err,
        marker=MARKERS["masked"],
        color=COLORS["masked"],
        label="Masked HNSW recall@K",
        linewidth=2.2,
    )
    _errorbar(
        ax_recall,
        ratios,
        rescore_recall,
        rescore_rec_err,
        marker=MARKERS["rescore"],
        color=COLORS["rescore"],
        label="Focus Rescore recall@K",
        linewidth=2.2,
    )
    ax_recall.set_xlabel("Subspace Ratio (D_sub / D)")
    ax_recall.set_ylabel("Recall@K (subspace GT, mean ± std)")
    ax_recall.set_ylim(0.0, 1.05)
    ax_recall.set_xticks(ratios)
    ax_recall.legend(loc="lower left")
    ax_recall.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    fig.tight_layout()

    ensure_output_dirs()
    png_path = bench_common.PLOTS_DIR / f"{output_stem}_d{dimension}.png"
    pdf_path = bench_common.PLOTS_DIR / f"{output_stem}_d{dimension}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_subspace_coherence(dimension: int, output_stem: str = "plot6_subspace_coherence") -> Path | None:
    """
    Plot 6: Subspace coherence — mean overlap between full-space and subspace
    exact top-k neighbour sets vs subspace ratio. Explains the Test E recall
    curves (a coherent subspace preserves neighbourhoods; an incoherent one
    cannot be searched accurately regardless of traversal).
    """
    rows = _read_csv(bench_common.RESULTS_DIR / f"subspace_coherence_d{dimension}.csv")
    if not rows:
        return None

    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    ratios = [float(r["subspace_ratio"]) for r in rows]
    overlap_cols = sorted(
        (c for c in rows[0] if c.startswith("overlap_at_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    palette = [COLORS["masked"], COLORS["rescore"], COLORS["speedup"], COLORS["vanilla"]]
    for i, col in enumerate(overlap_cols):
        k = col.rsplit("_", 1)[1]
        _errorbar(
            ax,
            ratios,
            _col(rows, col),
            _col(rows, f"{col}_std"),
            marker="o",
            color=palette[i % len(palette)],
            label=f"Overlap@{k}",
            linewidth=2.2,
        )

    ax.set_xlabel("Subspace Ratio (D_sub / D)")
    ax.set_ylabel("Full-space ∩ subspace top-k / k (mean ± std)")
    ax.set_title(f"Subspace Neighbourhood Coherence (D={dimension})")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(ratios)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ensure_output_dirs()
    png_path = bench_common.PLOTS_DIR / f"{output_stem}_d{dimension}.png"
    pdf_path = bench_common.PLOTS_DIR / f"{output_stem}_d{dimension}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def generate_all_plots(dimensions: list[int]) -> list[Path]:
    """Render all publication charts for each evaluated dimension."""
    created: list[Path] = []
    for dim in dimensions:
        for plotter in (
            plot_latency_recall,
            plot_throughput,
            plot_attribution_depth,
            plot_subspace_speedup,
            plot_masked_subspace,
            plot_subspace_coherence,
        ):
            path = plotter(dim)
            if path is not None:
                created.append(path)
    return created
