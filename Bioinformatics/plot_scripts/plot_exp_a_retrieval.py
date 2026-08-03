"""Experiment A — retrieval Recall@k bar chart (XQdrant vs baselines)."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BIO_ROOT, PALETTE, RESULTS, read_tsv, save, setup_style
METHOD_LABELS = {'xqdrant_esm2': 'XQdrant', 'esm2_cosine': 'ESM2 cosine', 'blastp': 'BLASTp', 'foldseek': 'Foldseek', 'tmalign_rerank_foldseek': 'TM-align†'}
METHOD_COLORS = {'xqdrant_esm2': PALETTE['xqdrant'], 'esm2_cosine': PALETTE['esm2'], 'blastp': PALETTE['blast'], 'foldseek': PALETTE['foldseek'], 'tmalign_rerank_foldseek': PALETTE['tmalign']}
RECALL_ALPHA = {'R@1': 0.55, 'R@5': 0.75, 'R@10': 1.0}

def main() -> int:
    setup_style()
    tsv = RESULTS / 'exp_a' / 'baselines' / 'comparison.tsv'
    if not tsv.exists():
        print(f'ERROR: missing {tsv}', file=sys.stderr)
        return 1
    rows = read_tsv(tsv)
    methods = [r['method'] for r in rows]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    r1 = [float(r['fold_R@1']) for r in rows]
    r5 = [float(r['fold_R@5']) for r in rows]
    r10 = [float(r['fold_R@10']) for r in rows]
    x = np.arange(len(methods))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for vals, lab, off in [(r1, 'R@1', -w), (r5, 'R@5', 0), (r10, 'R@10', w)]:
        alpha = RECALL_ALPHA[lab]
        for xi, m, v in zip(x + off, methods, vals):
            ax.bar(xi, v, width=w, color=METHOD_COLORS.get(m, '#64748b'), alpha=alpha, edgecolor='white', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Fold-level recall')
    ax.set_ylim(0, 1.05)
    method_handles = [mpatches.Patch(facecolor=METHOD_COLORS[m], edgecolor='white', label=METHOD_LABELS.get(m, m)) for m in methods]
    recall_handles = [mpatches.Patch(facecolor=PALETTE['xqdrant'], alpha=a, edgecolor='white', label=lab) for lab, a in RECALL_ALPHA.items()]
    leg1 = ax.legend(handles=method_handles, title='Method', loc='lower left', frameon=True, fontsize=8, title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=recall_handles, title='Metric', loc='lower right', frameon=True, fontsize=8, title_fontsize=8)
    ax.set_title('Experiment A - retrieval accuracy (n=1500 holdout queries)')
    fig.subplots_adjust(bottom=0.24)
    ax.text(
        0.0, -0.28, '† TM-align rerank on Foldseek top hits',
        transform=ax.transAxes, ha='left', va='top',
        fontsize=8, color='#64748b', clip_on=False,
    )
    save(fig, 'exp_a_retrieval_recall')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
