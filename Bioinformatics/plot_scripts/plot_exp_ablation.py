"""Plot layer/pooling ablation — fold-gap and Mann–Whitney p across configs."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUTDIR, RESULTS, ensure_out, save, setup_style

def main() -> int:
    setup_style()
    ensure_out()
    path = RESULTS / 'exp_ablation' / 'metrics.json'
    if not path.exists():
        print('ERROR: run ./run_exp_ablation.sh first', file=sys.stderr)
        return 1
    data = json.loads(path.read_text())
    configs = [(k, v) for k, v in sorted(data.get('configs', {}).items()) if not v.get('skipped') and v.get('fold_gap') is not None]
    if not configs:
        print('ERROR: no scored ablation configs', file=sys.stderr)
        return 1
    labels = [k.replace('_', '\n') for k, _ in configs]
    gaps = [float(v['fold_gap']) for _, v in configs]
    mwp = [-np.log10(max(float(v['mann_whitney_p']), 1e-300)) for _, v in configs]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    x = np.arange(len(labels))
    axes[0].bar(x, gaps, color='#2563eb')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel('Same − diff fold Spearman gap')
    axes[0].set_title('Layer / pooling ablation')
    axes[1].bar(x, mwp, color='#059669')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel('$-\\log_{10}$ Mann–Whitney $p$')
    axes[1].axhline(-np.log10(0.05), color='#94a3b8', ls='--', lw=1)
    save(fig, 'exp_ablation_layer_pooling')
    plt.close(fig)
    print(f'Wrote {OUTDIR}/exp_ablation_layer_pooling.png')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
