#!/usr/bin/env python3
"""Experiment D — thermostability case study figures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BIO_ROOT, PALETTE, RESULTS, load_json, read_tsv, save, setup_style  # noqa: E402

FEATURE_LABELS = {
    "frac_charged_exposed": "Δ charged exposed",
    "mean_loop_length": "Δ loop length",
    "packing_density": "Δ packing (SASA)",
    "ion_pair_density": "Δ ion-pair density",
}


def scatter_attribution_vs_delta(metrics: dict) -> None:
    pairs = metrics["pairs"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes = axes.ravel()
    colors_by_source = {
        "literature": PALETTE["literature"],
        "structure_map": PALETTE["structure_map"],
        "auto_fold": PALETTE["auto_fold"],
    }

    for ax, (key, ylab) in zip(axes, FEATURE_LABELS.items()):
        xs, ys, cs = [], [], []
        for p in pairs:
            fd = p.get("features_delta") or {}
            if p.get("xq_profile_spearman") is None or fd.get(key) is None:
                continue
            xs.append(float(p["xq_profile_spearman"]))
            ys.append(float(fd[key]))
            cs.append(colors_by_source.get(p.get("source", ""), "#64748b"))
        ax.scatter(xs, ys, c=cs, alpha=0.75, edgecolors="white", linewidths=0.4, s=42)
        if len(xs) >= 3:
            coef = np.polyfit(xs, ys, 1)
            xl = np.linspace(min(xs), max(xs), 50)
            ax.plot(xl, np.polyval(coef, xl), "--", color="#64748b", lw=1)
        rho = metrics.get("attribution_vs_delta_spearman", {}).get(key)
        pval = metrics.get("attribution_vs_delta_p_value", {}).get(key)
        title = ylab
        if rho is not None and pval is not None:
            title += f"\nρ={rho:.2f}, p={pval:.3f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("XQ profile Spearman")
        ax.set_ylabel(ylab)
        ax.axhline(0, color="#e2e8f0", lw=0.7)
        ax.axvline(0, color="#e2e8f0", lw=0.7)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=k)
        for k, c in colors_by_source.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Experiment D — attribution vs thermostability deltas (n={metrics['n_pairs']})", y=1.06)
    fig.tight_layout()
    save(fig, "exp_d_scatter_attribution_vs_delta")


def forest_significance(significance_rows: list[dict]) -> None:
    labels, rhos, lo, hi, sig = [], [], [], [], []
    for row in significance_rows:
        labels.append(FEATURE_LABELS.get(row["feature_delta"], row["feature_delta"]))
        rhos.append(float(row["spearman_rho"]))
        lo.append(float(row["ci95_lo"]))
        hi.append(float(row["ci95_hi"]))
        sig.append(row.get("significant_0_05") == "True")

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, (r, l, h, s) in enumerate(zip(rhos, lo, hi, sig)):
        color = PALETTE["xqdrant"] if s else PALETTE["diff_fold"]
        ax.plot([l, h], [i, i], color=color, lw=2)
        ax.plot(r, i, "o", color=color, ms=9)
    ax.axvline(0, color="#cbd5e1", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Spearman ρ (attribution vs Δ feature)")
    ax.set_title("Experiment D — cross-pair correlation + 95% CI")
    save(fig, "exp_d_forest_significance")


def source_breakdown(metrics: dict) -> None:
    by_src = metrics.get("by_source", {})
    names = list(by_src.keys())
    ns = [by_src[s]["n_pairs"] for s in names]
    spears = [by_src[s].get("mean_xq_profile_spearman") or 0 for s in names]
    colors = [PALETTE.get(s, PALETTE["auto_fold"]) for s in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    ax1.bar(names, ns, color=colors, width=0.55)
    ax1.set_ylabel("Pair count")
    ax1.set_title("Pairs by source")
    ax2.bar(names, spears, color=colors, width=0.55)
    ax2.set_ylabel("Mean XQ Spearman")
    ax2.set_title("Attribution score by source")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Experiment D — n={metrics['n_pairs']} meso/thermo pairs")
    fig.tight_layout()
    save(fig, "exp_d_source_breakdown")


def delta_feature_means(metrics: dict) -> None:
    deltas = metrics.get("mean_feature_delta", {})
    keys = list(FEATURE_LABELS.keys())
    vals = [deltas.get(k, 0) for k in keys]
    labels = [FEATURE_LABELS[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    colors = [PALETTE["pos"] if v >= 0 else PALETTE["neg"] for v in vals]
    ax.barh(labels, vals, color=colors, height=0.55)
    ax.axvline(0, color="#cbd5e1", lw=1)
    ax.set_xlabel("Mean thermo − meso")
    ax.set_title("Experiment D — aggregate structural deltas")
    save(fig, "exp_d_mean_deltas")


def main() -> int:
    setup_style()
    metrics_path = RESULTS / "exp_d" / "metrics.json"
    sig_path = RESULTS / "exp_d" / "significance.tsv"
    if not metrics_path.exists():
        print("ERROR: run Exp D first (results/exp_d/)", file=sys.stderr)
        return 1

    metrics = load_json(metrics_path)
    scatter_attribution_vs_delta(metrics)
    if sig_path.exists():
        forest_significance(read_tsv(sig_path))
    source_breakdown(metrics)
    delta_feature_means(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
