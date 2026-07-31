"""Exp A baseline: BLASTp holdout → train DB."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import ensure_dir, load_yaml, resolve_path, write_json
from exp_a_metrics_lib import evaluate_rankings, load_chain_meta

def which_or_die(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise FileNotFoundError(f'{name} not on PATH. source tools/env_linux.sh / micromamba bio-tools')
    return p

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--restart', action='store_true')
    ap.add_argument('--topk', type=int, default=10)
    args = ap.parse_args()
    bcfg = load_yaml('exp_a_baselines.yaml')
    blast_cfg = bcfg.get('blast') or {}
    corpus_cfg = load_yaml('corpus.yaml')
    smoke = bool(args.smoke)
    base = resolve_path('data/baselines/exp_a_smoke' if smoke else bcfg['baselines_dir'])
    out_dir = ensure_dir(resolve_path(bcfg['smoke_results_dir'] if smoke else bcfg['results_dir']) / 'blastp')
    train_fa = base / 'fasta' / 'train.fasta'
    hold_fa = base / 'fasta' / 'holdout.fasta'
    if not train_fa.exists() or not hold_fa.exists():
        print('ERROR: missing FASTA. Run prep_exp_a_baselines.py first.', file=sys.stderr)
        return 1
    db_dir = ensure_dir(base / 'blast_db')
    db_prefix = db_dir / 'train'
    hits_path = out_dir / 'blast_hits.tsv'
    rankings_path = out_dir / 'rankings.json'
    makeblastdb = which_or_die('makeblastdb')
    blastp = which_or_die('blastp')
    t0 = time.time()
    db_ready = (db_dir / 'train.pin').exists() or (db_dir / 'train.pdb').exists()
    if args.restart or not db_ready:
        for p in db_dir.glob('train.*'):
            p.unlink()
        print('Building BLAST DB...')
        subprocess.run([makeblastdb, '-in', str(train_fa), '-dbtype', 'prot', '-out', str(db_prefix), '-title', 'exp_a_train'], check=True)
    if args.restart or not hits_path.exists():
        print('Running blastp...')
        cmd = [blastp, '-query', str(hold_fa), '-db', str(db_prefix), '-outfmt', '6 qseqid sseqid evalue bitscore pident', '-evalue', str(blast_cfg.get('evalue', 10)), '-max_target_seqs', str(blast_cfg.get('max_target_seqs', 50)), '-num_threads', str(blast_cfg.get('num_threads', 4)), '-out', str(hits_path)]
        subprocess.run(cmd, check=True)
    else:
        print(f'Resume: using {hits_path}')
    by_q: dict[str, list[tuple[float, str]]] = defaultdict(list)
    with hits_path.open() as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            q, s, _e, bits = (parts[0], parts[1], parts[2], float(parts[3]))
            if q == s:
                continue
            by_q[q].append((bits, s))
    topk = args.topk
    rankings: dict[str, list[str]] = {}
    hold_ids = []
    with hold_fa.open() as f:
        for line in f:
            if line.startswith('>'):
                hold_ids.append(line[1:].strip().split()[0])
    for q in hold_ids:
        pairs = sorted(by_q.get(q, []), key=lambda x: -x[0])
        seen = set()
        hits = []
        for _b, s in pairs:
            if s in seen:
                continue
            seen.add(s)
            hits.append(s)
            if len(hits) >= topk:
                break
        rankings[q] = hits
    write_json(rankings_path, rankings)
    corpus_dir = resolve_path(corpus_cfg['smoke_outdir'] if smoke else corpus_cfg['paper_outdir'])
    meta = load_chain_meta(corpus_dir / 'chains.csv')
    ks = list(bcfg.get('limits') or [1, 5, 10])
    metrics = evaluate_rankings(hold_ids, rankings, meta, ks, n_boot=int(bcfg.get('bootstrap', 1000)), seed=int(bcfg.get('bootstrap_seed', 42)))
    metrics['method'] = 'blastp'
    metrics['mode'] = 'smoke' if smoke else 'paper'
    metrics['elapsed_s'] = round(time.time() - t0, 1)
    write_json(out_dir / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
