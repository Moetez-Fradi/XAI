"""Experiment B v3 — attribution validity figures."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BIO_ROOT, PALETTE, RESULTS, load_json, read_tsv, save, setup_style

def violin_same_vs_diff(pairs_path: Path, out_name: str) -> None:
    data = load_json(pairs_path)
    pairs = data.get('pairs') or data
    same = [float(p['spearman']) for p in pairs if p.get('same_fold')]
    diff = [float(p['spearman']) for p in pairs if not p.get('same_fold')]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    parts = ax.violinplot([same, diff], positions=[1, 2], showmeans=True, showmedians=True, widths=0.75)
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(PALETTE['same_fold'] if i == 0 else PALETTE['diff_fold'])
        body.set_alpha(0.75)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Same fold\n(n={len(same)})', f'Diff fold\n(n={len(diff)})'])
    ax.set_ylabel('XQ profile Spearman')
    ax.axhline(0, color='#e2e8f0', lw=0.8)
    ax.set_title('Experiment B - attribution score by fold label')
    save(fig, out_name)

def label_perm_bar(summary_rows: list[dict]) -> None:
    primary = {r['metric']: r['value'] for r in summary_rows if r.get('section') == 'primary'}
    real = float(primary['real_label_perm_fold_gap'])
    null = float(primary['null_mean_label_perm_fold_gap'])
    p = float(primary['p_value_label_perm_fold_gap'])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(['Null (label perm)', 'Observed'], [null, real], color=[PALETTE['null'], PALETTE['xqdrant']], width=0.55)
    ax.set_ylabel('Same − diff fold mean Spearman')
    ax.set_title(f'Label-permutation specificity\np = {p:.4f}')
    save(fig, 'exp_b_v3_label_perm')

def ablation_topn(summary_rows: list[dict]) -> None:
    tops = []
    spears = []
    for r in summary_rows:
        sec = r.get('section', '')
        if sec.startswith('ablation.top_') and r.get('metric') == 'mean_spearman':
            n = sec.split('top_')[1]
            tops.append(int(n))
            spears.append(float(r['value']))
    order = np.argsort(tops)
    tops = [tops[i] for i in order]
    spears = [spears[i] for i in order]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(tops, spears, 'o-', color=PALETTE['xqdrant'], lw=2, ms=8)
    ax.set_xlabel('Top-N attributed dimensions')
    ax.set_ylabel('Mean profile Spearman')
    ax.set_title('Ablation - attribution dimension count')
    ax.set_xticks(tops)
    save(fig, 'exp_b_v3_ablation_topn')

def mann_whitney_inset(summary_rows: list[dict]) -> None:
    primary = {r['metric']: r['value'] for r in summary_rows if r.get('section') == 'primary'}
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(['Same fold', 'Diff fold'], [float(primary['mean_spearman_same_fold']), float(primary['mean_spearman_diff_fold'])], color=[PALETTE['same_fold'], PALETTE['diff_fold']], width=0.55)
    p = float(primary['mann_whitney_p'])
    ax.set_ylabel('Mean profile Spearman')
    ax.set_title(f'Same vs different SCOP fold\nMann–Whitney p ≈ {p:.1e}')
    save(fig, 'exp_b_v3_same_vs_diff_means')

def main() -> int:
    setup_style()
    pairs_path = RESULTS / 'exp_b_v3' / 'pairs.json'
    summary_path = RESULTS / 'exp_b_v3' / 'summary.tsv'
    if not pairs_path.exists() or not summary_path.exists():
        print('ERROR: run Exp B v3 first (results/exp_b_v3/)', file=sys.stderr)
        return 1
    summary_rows = read_tsv(summary_path)
    violin_same_vs_diff(pairs_path, 'exp_b_v3_violin_spearman')
    label_perm_bar(summary_rows)
    ablation_topn(summary_rows)
    mann_whitney_inset(summary_rows)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
