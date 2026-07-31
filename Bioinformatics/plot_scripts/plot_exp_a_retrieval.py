"""Experiment A — retrieval Recall@k bar chart (XQdrant vs baselines)."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BIO_ROOT, PALETTE, RESULTS, read_tsv, save, setup_style
METHOD_LABELS = {'xqdrant_esm2': 'XQdrant', 'esm2_cosine': 'ESM2 cosine', 'blastp': 'BLASTp', 'foldseek': 'Foldseek', 'tmalign_rerank_foldseek': 'TM-align†'}
METHOD_COLORS = {'xqdrant_esm2': PALETTE['xqdrant'], 'esm2_cosine': PALETTE['esm2'], 'blastp': PALETTE['blast'], 'foldseek': PALETTE['foldseek'], 'tmalign_rerank_foldseek': PALETTE['tmalign']}

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
    for i, (vals, lab, off) in enumerate([(r1, 'R@1', -w), (r5, 'R@5', 0), (r10, 'R@10', w)]):
        colors = [METHOD_COLORS.get(m, '#64748b') for m in methods]
        bars = ax.bar(x + off, vals, width=w, label=lab, color=colors, alpha=0.85 if off == 0 else 0.65)
        if off == 0:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Fold-level recall')
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc='lower right', frameon=True)
    ax.set_title('Experiment A — retrieval accuracy (n=1500 holdout queries)')
    fig.text(0.01, 0.01, '† TM-align rerank on Foldseek top hits', fontsize=8, color='#64748b')
    save(fig, 'exp_a_retrieval_recall')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
