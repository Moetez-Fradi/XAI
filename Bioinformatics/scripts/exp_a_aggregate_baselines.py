"""Merge Exp A XQdrant + baseline metrics into one comparison table."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import ensure_dir, load_yaml, resolve_path, write_json
METHODS = [('xqdrant_esm2', 'results/exp_a/metrics.json', 'results/exp_a_smoke/metrics.json'), ('esm2_cosine', 'results/exp_a/baselines/esm2_cosine/metrics.json', 'results/exp_a_smoke/baselines/esm2_cosine/metrics.json'), ('blastp', 'results/exp_a/baselines/blastp/metrics.json', 'results/exp_a_smoke/baselines/blastp/metrics.json'), ('foldseek', 'results/exp_a/baselines/foldseek/metrics.json', 'results/exp_a_smoke/baselines/foldseek/metrics.json'), ('tmalign_rerank_foldseek', 'results/exp_a/baselines/tmalign/metrics.json', 'results/exp_a_smoke/baselines/tmalign/metrics.json')]

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    smoke = bool(args.smoke)
    bcfg = load_yaml('exp_a_baselines.yaml')
    out_dir = ensure_dir(resolve_path(bcfg['smoke_results_dir'] if smoke else bcfg['results_dir']))
    rows = []
    missing = []
    for name, paper_p, smoke_p in METHODS:
        path = resolve_path(smoke_p if smoke else paper_p)
        if not path.exists():
            missing.append(str(path))
            continue
        m = json.loads(path.read_text())
        fold = m.get('fold') or {}
        rows.append({'method': name, 'n_queries': m.get('n_queries'), 'fold_recall@1': fold.get('recall@1'), 'fold_recall@5': fold.get('recall@5'), 'fold_recall@10': fold.get('recall@10'), 'fold_precision@1': fold.get('precision@1'), 'family_recall@1': (m.get('family') or {}).get('recall@1'), 'attribution_rank_stable': m.get('attribution_rank_stable'), 'source': str(path)})
    write_json(out_dir / 'comparison.json', {'methods': rows, 'missing': missing})
    tsv = out_dir / 'comparison.tsv'
    with tsv.open('w') as f:
        f.write('method\tn_queries\tfold_R@1\tfold_R@5\tfold_R@10\tfamily_R@1\n')
        for r in rows:
            f.write(f"{r['method']}\t{r['n_queries']}\t{r['fold_recall@1']}\t{r['fold_recall@5']}\t{r['fold_recall@10']}\t{r['family_recall@1']}\n")
    print(tsv.read_text())
    if missing:
        print('Missing (run those baselines):', *missing, sep='\n  ')
    print(f"Wrote {out_dir / 'comparison.json'}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
