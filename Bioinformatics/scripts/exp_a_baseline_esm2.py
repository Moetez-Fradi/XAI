"""Exp A baseline: raw ESM2 cosine (no XQdrant / no attribution)."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import ensure_dir, load_yaml, resolve_path, write_json
from exp_a_metrics_lib import evaluate_rankings, load_chain_meta

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--restart', action='store_true')
    ap.add_argument('--topk', type=int, default=10)
    args = ap.parse_args()
    bcfg = load_yaml('exp_a_baselines.yaml')
    corpus_cfg = load_yaml('corpus.yaml')
    smoke = bool(args.smoke)
    out_dir = ensure_dir(resolve_path(bcfg['smoke_results_dir'] if smoke else bcfg['results_dir']) / 'esm2_cosine')
    ckpt = out_dir / 'rankings.json'
    if ckpt.exists() and (not args.restart):
        print(f'Resume: using existing {ckpt}')
        rankings = json.loads(ckpt.read_text())
    else:
        emb_dir = resolve_path('embeddings/smoke' if smoke else 'embeddings/esm2_t33_650M')
        corpus_dir = resolve_path(corpus_cfg['smoke_outdir'] if smoke else corpus_cfg['paper_outdir'])
        meta = load_chain_meta(corpus_dir / 'chains.csv')
        ids = [ln.strip() for ln in (emb_dir / 'ids.txt').read_text().splitlines() if ln.strip()]
        vec = np.load(emb_dir / 'vectors.npy').astype(np.float32)
        id_to_i = {c: i for i, c in enumerate(ids)}
        train_ids = [c for c in ids if meta.get(c, {}).get('split') == 'train']
        hold_ids = [c for c in ids if meta.get(c, {}).get('split') == 'holdout']
        hold_ids.sort()
        train_idx = np.array([id_to_i[c] for c in train_ids], dtype=np.int64)
        train_mat = vec[train_idx]
        train_mat = train_mat / (np.linalg.norm(train_mat, axis=1, keepdims=True) + 1e-12)
        topk = args.topk
        rankings = {}
        t0 = time.time()
        print(f'ESM2 cosine: {len(hold_ids)} queries × {len(train_ids)} train')
        for qi, qid in enumerate(hold_ids):
            q = vec[id_to_i[qid]]
            q = q / (np.linalg.norm(q) + 1e-12)
            scores = train_mat @ q
            order = np.argpartition(-scores, min(topk, len(scores) - 1))[:topk]
            order = order[np.argsort(-scores[order])]
            hits = []
            for j in order:
                hid = train_ids[int(j)]
                if hid == qid:
                    continue
                hits.append(hid)
                if len(hits) >= topk:
                    break
            rankings[qid] = hits
            if (qi + 1) % 100 == 0 or qi + 1 == len(hold_ids):
                print(f'  [{qi + 1}/{len(hold_ids)}] elapsed {time.time() - t0:.1f}s', flush=True)
        write_json(ckpt, rankings)
    corpus_dir = resolve_path(corpus_cfg['smoke_outdir'] if smoke else corpus_cfg['paper_outdir'])
    meta = load_chain_meta(corpus_dir / 'chains.csv')
    ks = list(bcfg.get('limits') or [1, 5, 10])
    qids = sorted(rankings.keys())
    metrics = evaluate_rankings(qids, rankings, meta, ks, n_boot=int(bcfg.get('bootstrap', 1000)), seed=int(bcfg.get('bootstrap_seed', 42)))
    metrics['method'] = 'esm2_cosine'
    metrics['mode'] = 'smoke' if smoke else 'paper'
    write_json(out_dir / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
