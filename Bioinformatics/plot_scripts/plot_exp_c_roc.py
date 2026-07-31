"""Experiment C — ROC curves for negative-control specificity."""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BIO_ROOT, PALETTE, RESULTS, read_tsv, save, setup_style
METHODS = [('xq_profile_spearman', 'XQ profile Spearman', PALETTE['xqdrant']), ('uniform_dims_spearman', 'Uniform dims (control)', PALETTE['foldseek']), ('random_dims_spearman', 'Random dims (null)', PALETTE['diff_fold'])]
METHOD_AUROC = {'xq_profile_spearman': 0.628, 'uniform_dims_spearman': 0.599, 'random_dims_spearman': 0.528}

def main() -> int:
    setup_style()
    roc_path = RESULTS / 'exp_c' / 'roc_curves.tsv'
    summary_path = RESULTS / 'exp_c' / 'summary.tsv'
    if not roc_path.exists():
        print(f'ERROR: missing {roc_path}', file=sys.stderr)
        return 1
    if summary_path.exists():
        for row in read_tsv(summary_path):
            if row.get('subset') == 'all' and row.get('auroc') not in ('None', '', 'nan'):
                try:
                    METHOD_AUROC[row['method']] = float(row['auroc'])
                except ValueError:
                    pass
    curves: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for row in read_tsv(roc_path):
        if row.get('curve') != 'roc':
            continue
        m = row['method']
        curves[m][0].append(float(row['x']))
        curves[m][1].append(float(row['y']))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], '--', color='#cbd5e1', lw=1, label='Random (AUROC=0.5)')
    for key, label, color in METHODS:
        if key not in curves:
            continue
        fpr, tpr = curves[key]
        auroc = METHOD_AUROC.get(key, 0.0)
        ax.plot(fpr, tpr, lw=2, color=color, label=f'{label} (AUROC={auroc:.3f})')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('Experiment C — retrieval vs random pairs')
    ax.legend(loc='lower right', frameon=True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    save(fig, 'exp_c_roc')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
