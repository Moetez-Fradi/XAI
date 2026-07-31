"""Experiment A — retrieval sanity + attribution non-degradation check.

For each holdout query, search train-only neighbors in XQdrant and score:
  - same SCOPe fold / family in top-k (Recall-style hit rates)
  - plain search vs with_dims_explained (top-k ids must match)

Artifacts under results/exp_a[/ _smoke]/:
  run_config.json
  query_list.txt
  checkpoints/queries.jsonl     # one record per finished query (resume)
  checkpoints/state.json
  per_query.json                # final aggregate
  metrics.json                  # + bootstrap CIs over queries
  summary.tsv

Resume from checkpoints unless --restart.
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import BIO_ROOT, BatchProgress, ensure_dir, load_yaml, resolve_path, write_json
from xqdrant_rest import XQdrantREST
from exp_a_metrics_lib import mean_average_precision

def load_meta(chains_csv: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with chains_csv.open() as f:
        for rec in csv.DictReader(f):
            out[rec['chain_id']] = rec
    return out

def family_key(sccs: str) -> str:
    return sccs or ''

def superfamily_key(sccs: str) -> str:
    bits = (sccs or '').split('.')
    return '.'.join(bits[:3]) if len(bits) >= 3 else sccs

def hit_rates(flags: list[list[bool]], ks: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    n = len(flags)
    if n == 0:
        for k in ks:
            out[f'recall@{k}'] = 0.0
            out[f'precision@{k}'] = 0.0
        return out
    for k in ks:
        hits = sum((1 for row in flags if any(row[:k])))
        out[f'recall@{k}'] = hits / n
        prec = [sum(row[:k]) / len(row[:k]) if row[:k] else 0.0 for row in flags]
        out[f'precision@{k}'] = float(np.mean(prec))
    return out

def bootstrap_cis(flags: list[list[bool]], ks: list[int], *, n_boot: int=1000, seed: int=42, alpha: float=0.05) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(flags)
    out: dict[str, dict[str, float]] = {}
    if n == 0:
        return out
    idx = np.arange(n)
    for k in ks:
        scores = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            sample = rng.choice(idx, size=n, replace=True)
            hits = sum((1 for i in sample if any(flags[i][:k])))
            scores[b] = hits / n
        lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
        out[f'recall@{k}'] = {'mean': float(scores.mean()), 'ci_low': float(lo), 'ci_high': float(hi), 'n_boot': n_boot, 'alpha': alpha}
    return out

def load_jsonl_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def append_jsonl(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open('a') as f:
        f.write(json.dumps(obj, sort_keys=True) + '\n')
        f.flush()

def record_to_flags(rec: dict) -> tuple[list[bool], list[bool], list[bool]]:
    neigh = rec.get('neighbors') or []
    return ([bool(n.get('same_fold')) for n in neigh], [bool(n.get('same_superfamily')) for n in neigh], [bool(n.get('same_family')) for n in neigh])

def finalize_metrics(records: list[dict], *, ks: list[int], coll: str, smoke: bool, dims_top: int, elapsed_s: float, n_boot: int, boot_seed: int) -> dict:
    fold_flags, sf_flags, fam_flags = ([], [], [])
    rank_mismatches = 0
    explain_present = 0
    for rec in records:
        f, sf, fam = record_to_flags(rec)
        fold_flags.append(f)
        sf_flags.append(sf)
        fam_flags.append(fam)
        if not rec.get('rank_match_with_explain', True):
            rank_mismatches += 1
        if any((n.get('dims_explained') for n in rec.get('neighbors') or [])):
            explain_present += 1
    return {'fold': hit_rates(fold_flags, ks), 'superfamily': hit_rates(sf_flags, ks), 'family': hit_rates(fam_flags, ks), 'fold_map': mean_average_precision(fold_flags), 'superfamily_map': mean_average_precision(sf_flags), 'family_map': mean_average_precision(fam_flags), 'fold_recall_bootstrap': bootstrap_cis(fold_flags, ks, n_boot=n_boot, seed=boot_seed), 'n_queries': len(records), 'rank_mismatches_plain_vs_explained': rank_mismatches, 'queries_with_dims_explained': explain_present, 'attribution_rank_stable': rank_mismatches == 0, 'elapsed_s': round(elapsed_s, 1), 'collection': coll, 'mode': 'smoke' if smoke else 'paper', 'dims_explained_top': dims_top, 'completed_unix': int(time.time())}

def write_summary_tsv(path: Path, metrics: dict) -> None:
    with path.open('w') as f:
        f.write('level\tmetric\tvalue\n')
        for level in ('fold', 'superfamily', 'family'):
            for k, v in metrics[level].items():
                f.write(f'{level}\t{k}\t{v:.6f}\n')
        for level in ('fold', 'superfamily', 'family'):
            mk = f'{level}_map'
            if mk in metrics:
                f.write(f'{level}\tmap\t{metrics[mk]:.6f}\n')
        for k, v in (metrics.get('fold_recall_bootstrap') or {}).items():
            f.write(f"fold_boot\t{k}_ci_low\t{v['ci_low']:.6f}\n")
            f.write(f"fold_boot\t{k}_ci_high\t{v['ci_high']:.6f}\n")
        f.write(f"meta\tattribution_rank_stable\t{int(metrics['attribution_rank_stable'])}\n")
        f.write(f"meta\tn_queries\t{metrics['n_queries']}\n")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--restart', action='store_true', help='Wipe results/checkpoints for this run and start over')
    ap.add_argument('--url', default=None)
    ap.add_argument('--collection', default=None)
    ap.add_argument('--max-queries', type=int, default=None)
    ap.add_argument('--topk', type=int, default=10, help='Neighbors to retrieve')
    ap.add_argument('--bootstrap', type=int, default=1000, help='Bootstrap resamples')
    ap.add_argument('--bootstrap-seed', type=int, default=42)
    args = ap.parse_args()
    xcfg = load_yaml('xqdrant.yaml')
    corpus_cfg = load_yaml('corpus.yaml')
    exp_cfg = xcfg.get('exp_a') or {}
    smoke = bool(args.smoke)
    url = args.url or xcfg.get('url', 'http://127.0.0.1:6333')
    coll = args.collection or (xcfg['smoke_collection'] if smoke else xcfg['collection'])
    emb_dir = resolve_path('embeddings/smoke' if smoke else 'embeddings/esm2_t33_650M')
    corpus_dir = resolve_path(corpus_cfg['smoke_outdir'] if smoke else corpus_cfg['paper_outdir'])
    results_dir = resolve_path(exp_cfg['smoke_results_dir'] if smoke else exp_cfg['results_dir'])
    ckpt_dir = results_dir / 'checkpoints'
    jsonl_path = ckpt_dir / 'queries.jsonl'
    state_path = ckpt_dir / 'state.json'
    if args.restart and results_dir.exists():
        print(f'--restart: clearing {results_dir}')
        shutil.rmtree(results_dir)
    ensure_dir(results_dir)
    ensure_dir(ckpt_dir)
    ids = [ln.strip() for ln in (emb_dir / 'ids.txt').read_text().splitlines() if ln.strip()]
    vectors = np.load(emb_dir / 'vectors.npy')
    id_to_row = {c: i for i, c in enumerate(ids)}
    meta = load_meta(corpus_dir / 'chains.csv')
    emb_meta = {}
    if (emb_dir / 'meta.json').exists():
        emb_meta = json.loads((emb_dir / 'meta.json').read_text())
    lock_path = corpus_dir / 'corpus_lock.json'
    index_manifest = resolve_path(f'data/processed/xqdrant/{coll}_manifest.json')
    query_split = exp_cfg.get('query_split', 'holdout')
    filter_split = exp_cfg.get('index_filter_split', 'train')
    ks = list(exp_cfg.get('limits') or [1, 5, 10])
    topk = max(args.topk, max(ks))
    dims_top = int(xcfg.get('dims_explained_top', 32))
    queries = [cid for cid, m in meta.items() if m.get('split') == query_split and cid in id_to_row]
    queries.sort()
    max_q = args.max_queries
    if max_q is None:
        max_q = int(exp_cfg.get('max_queries_smoke', 20)) if smoke else int(exp_cfg.get('max_queries_paper', 0))
    if max_q and max_q > 0:
        queries = queries[:max_q]
    if not queries:
        print(f'ERROR: no queries with split={query_split} in embeddings.', file=sys.stderr)
        return 1
    run_config = {'mode': 'smoke' if smoke else 'paper', 'url': url, 'collection': coll, 'query_split': query_split, 'index_filter_split': filter_split, 'topk': topk, 'limits': ks, 'dims_explained_top': dims_top, 'n_queries_planned': len(queries), 'embeddings': str(emb_dir.relative_to(BIO_ROOT)), 'embeddings_meta': emb_meta, 'corpus': str(corpus_dir.relative_to(BIO_ROOT)), 'corpus_lock': str(lock_path.relative_to(BIO_ROOT)) if lock_path.exists() else None, 'index_manifest': str(index_manifest.relative_to(BIO_ROOT)) if index_manifest.exists() else None, 'bootstrap': args.bootstrap, 'bootstrap_seed': args.bootstrap_seed, 'started_unix': int(time.time())}
    write_json(results_dir / 'run_config.json', run_config)
    (results_dir / 'query_list.txt').write_text('\n'.join(queries) + '\n')
    done_records = load_jsonl_records(jsonl_path)
    done_ids = {r['query'] for r in done_records if 'query' in r}
    pending = [q for q in queries if q not in done_ids]
    if done_ids:
        print(f'Resume: {len(done_ids)} queries cached, {len(pending)} remaining')
    if not pending and done_records:
        print('All planned queries already checkpointed — recomputing metrics only.')
    client = XQdrantREST(url)
    if pending:
        try:
            client.wait_ready(retries=10, sleep_s=0.3)
        except RuntimeError as e:
            print(f'ERROR: {e}\nStart: ./start_xqdrant.sh', file=sys.stderr)
            return 1
        if not client.collection_exists(coll):
            print(f"ERROR: collection {coll} missing. Run index_xqdrant.py{(' --smoke' if smoke else '')} first.", file=sys.stderr)
            return 1
    qfilter = {'must': [{'key': 'split', 'match': {'value': filter_split}}]}
    print(f'Exp A | collection={coll} queries={len(queries)} (split={query_split}) filter={filter_split} topk={topk}')
    t0 = time.time()
    progress = BatchProgress(max(len(pending), 1), label='expA-query')
    if not pending:
        progress.tick(skipped=True, force_print=True)
    for qi, qid in enumerate(pending):
        qmeta = meta[qid]
        qvec = vectors[id_to_row[qid]].astype(np.float32).tolist()
        qfold = qmeta.get('scop_fold', '')
        qsccs = qmeta.get('scop_sccs', '')
        qfam = family_key(qsccs)
        qsf = superfamily_key(qsccs)
        t_q = time.time()
        plain = client.query(coll, qvec, limit=topk, with_payload=True, with_dims_explained=None, query_filter=qfilter)
        explained = client.query(coll, qvec, limit=topk, with_payload=True, with_dims_explained={'top': dims_top}, query_filter=qfilter)
        plain_ids = [str(h.get('payload', {}).get('chain_id', h['id'])) for h in plain]
        expl_ids = [str(h.get('payload', {}).get('chain_id', h['id'])) for h in explained]
        plain_scores = [h.get('score') for h in plain]
        expl_scores = [h.get('score') for h in explained]
        neigh_rows = []
        for rank, h in enumerate(plain):
            pl = h.get('payload') or {}
            cid = str(pl.get('chain_id', h['id']))
            same_f = bool(qfold) and pl.get('scop_fold') == qfold
            same_fam = bool(qfam) and family_key(pl.get('scop_sccs', '')) == qfam
            same_sf = bool(qsf) and superfamily_key(pl.get('scop_sccs', '')) == qsf
            dims = explained[rank].get('dims_explained') if rank < len(explained) else None
            neigh_rows.append({'rank': rank, 'point_id': h.get('id'), 'chain_id': cid, 'score': h.get('score'), 'score_with_explain': explained[rank].get('score') if rank < len(explained) else None, 'scop_fold': pl.get('scop_fold'), 'scop_sccs': pl.get('scop_sccs'), 'scop_class': pl.get('scop_class'), 'uniprot': pl.get('uniprot'), 'same_fold': same_f, 'same_family': same_fam, 'same_superfamily': same_sf, 'dims_explained': dims})
        rec = {'query': qid, 'query_point_id': id_to_row[qid], 'pdb_id': qmeta.get('pdb_id'), 'chain': qmeta.get('chain'), 'length': qmeta.get('length'), 'uniprot': qmeta.get('uniprot'), 'scop_domain': qmeta.get('scop_domain'), 'scop_fold': qfold, 'scop_sccs': qsccs, 'scop_class': qmeta.get('scop_class'), 'plain_neighbor_ids': plain_ids, 'explained_neighbor_ids': expl_ids, 'plain_scores': plain_scores, 'explained_scores': expl_scores, 'neighbors': neigh_rows, 'rank_match_with_explain': plain_ids == expl_ids, 'elapsed_s': round(time.time() - t_q, 4), 'unix': int(time.time())}
        append_jsonl(jsonl_path, rec)
        done_records.append(rec)
        write_json(state_path, {'n_done': len(done_records), 'n_planned': len(queries), 'last_query': qid, 'updated_unix': int(time.time())})
        if len(done_records) % 25 == 0 or qi + 1 >= len(pending):
            partial = finalize_metrics(done_records, ks=ks, coll=coll, smoke=smoke, dims_top=dims_top, elapsed_s=time.time() - t0, n_boot=min(args.bootstrap, 200), boot_seed=args.bootstrap_seed)
            partial['partial'] = True
            write_json(results_dir / 'metrics.partial.json', partial)
        progress.tick(ok=True, force_print=qi + 1 >= len(pending))
    by_q = {r['query']: r for r in done_records}
    ordered = [by_q[q] for q in queries if q in by_q]
    metrics = finalize_metrics(ordered, ks=ks, coll=coll, smoke=smoke, dims_top=dims_top, elapsed_s=time.time() - t0, n_boot=args.bootstrap, boot_seed=args.bootstrap_seed)
    metrics['partial'] = False
    write_json(results_dir / 'metrics.json', metrics)
    write_json(results_dir / 'per_query.json', {'queries': ordered})
    write_summary_tsv(results_dir / 'summary.tsv', metrics)
    write_json(state_path, {'n_done': len(ordered), 'n_planned': len(queries), 'complete': len(ordered) == len(queries), 'updated_unix': int(time.time())})
    print(json.dumps(metrics, indent=2))
    print(f'Wrote {results_dir.relative_to(BIO_ROOT)}/')
    print(f'  checkpoint: {jsonl_path.relative_to(BIO_ROOT)} ({len(ordered)} lines)')
    if metrics['rank_mismatches_plain_vs_explained']:
        print(f"WARN: {metrics['rank_mismatches_plain_vs_explained']} queries had rank changes with dims_explained", file=sys.stderr)
        return 2
    if metrics['queries_with_dims_explained'] == 0:
        print('ERROR: no dims_explained in responses — is this the XQdrant fork?', file=sys.stderr)
        return 1
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
