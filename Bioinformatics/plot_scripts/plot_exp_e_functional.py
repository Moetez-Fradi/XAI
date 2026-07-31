#!/usr/bin/env python3
"""Experiment E figures — XQ vs TM-align site enrichment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RESULTS, ensure_out, save  # noqa: E402


def load_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        rows.append({header[i]: parts[i] if i < len(parts) else "" for i in range(len(header))})
    return rows


def main() -> int:
    ensure_out()
    metrics_path = RESULTS / "exp_e" / "metrics.json"
    loc_path = RESULTS / "exp_e" / "localization_comparison.tsv"
    if not metrics_path.exists():
        print("ERROR: run Exp E first (results/exp_e/)", file=sys.stderr)
        return 1

    metrics = json.loads(metrics_path.read_text())
    rows = load_tsv(loc_path)

    func = [r for r in rows if r.get("pair_label") == "functional_match"]
    fold = [r for r in rows if r.get("pair_label") == "fold_only"]

    def vals(rs, key):
        out = []
        for r in rs:
            try:
                v = float(r.get(key) or "nan")
            except ValueError:
                continue
            if np.isfinite(v):
                out.append(v)
        return out

    xq_f, xq_o = vals(func, "xq_enrichment_ratio"), vals(fold, "xq_enrichment_ratio")
    tm_f, tm_o = vals(func, "tm_site_enrichment_ratio"), vals(fold, "tm_site_enrichment_ratio")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    data = [xq_f, xq_o, tm_f, tm_o]
    labels = ["XQ\nfunctional", "XQ\nfold-only", "TM\nfunctional", "TM\nfold-only"]
    colors = ["#2166ac", "#92c5de", "#b2182b", "#f4a582"]
    bp = axes[0].boxplot(
        [d if d else [0] for d in data],
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    axes[0].axhline(1.0, color="gray", ls="--", lw=0.8)
    axes[0].set_ylabel("Site enrichment ratio (>1 = localized)")
    axes[0].set_title("Exp E: active-site localization")

    # Scatter XQ vs TM (functional pairs)
    xf, yf = vals(func, "xq_enrichment_ratio"), vals(func, "tm_site_enrichment_ratio")
    n = min(len(xf), len(yf))
    if n > 0:
        axes[1].scatter(xf[:n], yf[:n], alpha=0.65, c="#2166ac", edgecolors="none")
        axes[1].axhline(1, color="gray", ls="--", lw=0.8)
        axes[1].axvline(1, color="gray", ls="--", lw=0.8)
        axes[1].set_xlabel("XQ attribution site enrichment")
        axes[1].set_ylabel("TM-align site overlap enrichment")
        axes[1].set_title(f"Functional pairs (n={n})")
    else:
        axes[1].text(0.5, 0.5, "No TM-align pairs", ha="center", va="center", transform=axes[1].transAxes)

    sig = metrics.get("significance") or {}
    fig.suptitle(
        f"Exp E | MW p={sig.get('mann_whitney_site_vs_background_p', '?'):.3g} "
        f"| XQ func mean={metrics.get('xq_site_enrichment', {}).get('functional_match_mean', '?')}",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "exp_e_site_enrichment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
