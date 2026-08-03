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
    pvals = [float(v['mann_whitney_p']) for _, v in configs]
    gap_colors = ['#059669' if g >= 0 else '#dc2626' for g in gaps]
    sig_colors = ['#059669' if p < 0.05 else '#94a3b8' for p in pvals]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    x = np.arange(len(labels))
    gap_bars = axes[0].bar(x, gaps, color=gap_colors, width=0.55)
    axes[0].axhline(0, color='#cbd5e1', lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel('Same − diff fold Spearman gap')
    axes[0].set_title('Layer / pooling ablation')
    for bar, val in zip(gap_bars, gaps):
        y = val + (0.004 if val >= 0 else -0.004)
        va = 'bottom' if val >= 0 else 'top'
        axes[0].text(bar.get_x() + bar.get_width() / 2, y, f'{val:+.3f}', ha='center', va=va, fontsize=8)
    p_bars = axes[1].bar(x, [-np.log10(max(p, 1e-300)) for p in pvals], color=sig_colors, width=0.55)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel('$-\\log_{10}$ Mann–Whitney $p$')
    axes[1].axhline(-np.log10(0.05), color='#64748b', ls='--', lw=1)
    for bar, p in zip(p_bars, pvals):
        h = bar.get_height()
        label = f'p={p:.1e}' if p < 0.05 else 'ns'
        y = h + 0.08 if p < 0.05 else 0.08
        axes[1].text(bar.get_x() + bar.get_width() / 2, y, label, ha='center', va='bottom', fontsize=7)
    save(fig, 'exp_ablation_layer_pooling')
    plt.close(fig)
    print(f'Wrote {OUTDIR}/exp_ablation_layer_pooling.png')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
