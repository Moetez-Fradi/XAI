"""Experiment F — runtime / workflow comparison (steps.md §4F).

For each holdout query, times three analyst workflows:

1. **XQdrant** — (optional) ESM2 embed + attributed vector search
2. **Foldseek → TM-align → PyMOL** — structure search, re-rank, manual inspection
3. **BLAST → TM-align → PyMOL** — sequence search, re-rank, manual inspection

Reports per-query wall-clock times, automated vs manual step counts, and one-time
setup costs (index / DB build) when measurable.

Artifacts:
  results/exp_f[/ _smoke]/
    run_config.json
    setup_costs.json
    per_query_timing.jsonl
    workflow_comparison.tsv
    summary.tsv
    metrics.json
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import BIO_ROOT, BatchProgress, ensure_dir, load_yaml, resolve_path, write_json
from exp_b_attribution import load_embeddings
from exp_f_lib import ESM2SingleEmbedder, load_fasta, summarize_times, time_blast_query, time_foldseek_query, time_tmalign_candidates, time_xqdrant_query, which_or_die, which_optional, which_tmalign
from xqdrant_rest import XQdrantREST

def load_meta(chains_csv: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with chains_csv.open() as f:
        for rec in csv.DictReader(f):
            out[rec['chain_id']] = rec
    return out

def train_only_filter() -> dict:
    return {'must': [{'key': 'split', 'match': {'value': 'train'}}]}

def select_queries(meta: dict[str, dict], *, id_to_row: dict[str, int], struct_dir: Path, fasta: dict[str, str], n: int, seed: int) -> list[str]:
    pool = []
    for cid, m in meta.items():
        if m.get('split') != 'holdout':
            continue
        if cid not in id_to_row:
            continue
        if cid not in fasta:
            continue
        if not (struct_dir / f'{cid}.pdb').exists():
            continue
        pool.append(cid)
    pool.sort()
    rng = random.Random(seed)
    if len(pool) <= n:
        return pool
    return sorted(rng.sample(pool, n))

def measure_setup_costs(*, base: Path, foldseek: str, makeblastdb: str | None, train_fa: Path, train_struct_dir: Path) -> dict:
    costs: dict[str, dict] = {}
    db_dir = base / 'foldseek_db'
    target_db = db_dir / 'train'
    db_ready = (db_dir / 'train.dbtype').exists() or (db_dir / 'train_ss.dbtype').exists()
    if db_ready:
        costs['foldseek_db'] = {'elapsed_s': 0.0, 'note': 'pre-built (amortized; not included in per-query times)'}
    elif train_struct_dir.exists() and any(train_struct_dir.glob('*.pdb')):
        ensure_dir(db_dir)
        t0 = time.perf_counter()
        subprocess.run([foldseek, 'createdb', str(train_struct_dir), str(target_db)], check=True, capture_output=True)
        costs['foldseek_db'] = {'elapsed_s': round(time.perf_counter() - t0, 1), 'note': 'measured during Exp F (one-time)'}
    else:
        costs['foldseek_db'] = {'elapsed_s': float('nan'), 'note': 'missing train structures'}
    blast_dir = ensure_dir(base / 'blast_db')
    db_prefix = blast_dir / 'train'
    blast_ready = (blast_dir / 'train.pin').exists() or (blast_dir / 'train.pdb').exists()
    if not makeblastdb:
        costs['blast_db'] = {'elapsed_s': float('nan'), 'note': 'BLAST+ not installed — workflow skipped'}
    elif blast_ready:
        costs['blast_db'] = {'elapsed_s': 0.0, 'note': 'pre-built (amortized)'}
    elif train_fa.exists():
        t0 = time.perf_counter()
        subprocess.run([makeblastdb, '-in', str(train_fa), '-dbtype', 'prot', '-out', str(db_prefix)], check=True, capture_output=True)
        costs['blast_db'] = {'elapsed_s': round(time.perf_counter() - t0, 1), 'note': 'measured during Exp F (one-time)'}
    else:
        costs['blast_db'] = {'elapsed_s': float('nan'), 'note': 'missing train.fasta'}
    xcfg = load_yaml('xqdrant.yaml')
    coll = xcfg.get('collection', 'proteins_esm2_650m')
    manifest = resolve_path(f'data/processed/xqdrant/{coll}_manifest.json')
    if manifest.exists():
        man = json.loads(manifest.read_text())
        costs['xqdrant_index'] = {'elapsed_s': man.get('elapsed_s'), 'n_points': man.get('n_points'), 'note': 'from index_xqdrant manifest (one-time corpus index)'}
    else:
        costs['xqdrant_index'] = {'elapsed_s': float('nan'), 'note': 'run index_xqdrant.py first'}
    return costs

def write_workflow_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter='\t')
        w.writeheader()
        for r in rows:
            w.writerow(r)

def run_indexed_pass(*, results_dir: Path, emb_dir: Path, id_to_row: dict[str, int], emb: np.ndarray, queries: list[str], client: XQdrantREST, coll: str, top_k: int, dims_top: int, q_filter: dict, restart: bool) -> int:
    ckpt = results_dir / 'per_query_timing_indexed.jsonl'
    if restart and ckpt.exists():
        ckpt.unlink()
    done: set[str] = set()
    if ckpt.exists():
        with ckpt.open() as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)['query'])
    pending = [q for q in queries if q not in done]
    print(f'Exp F indexed pass: {len(pending)} pending / {len(queries)} queries (search-only)')
    progress = BatchProgress(max(len(pending), 1), label='exp-f-idx')
    for qid in pending:
        row = id_to_row[qid]
        vec = emb[row].astype(np.float64).tolist()
        search_t, n_attr = time_xqdrant_query(client, collection=coll, vector=vec, limit=top_k, dims_top=dims_top, train_filter=q_filter)
        rec = {'query': qid, 'xqdrant_indexed': {'search_s': round(search_t, 4), 'n_hits_with_attribution': n_attr, 'total_automated_s': round(search_t, 4), 'automated_steps': 1, 'manual_steps': 0, 'delivers_rationale': 'automatic (dims_explained; pre-indexed embed)'}}
        with ckpt.open('a') as f:
            f.write(json.dumps(rec, sort_keys=True) + '\n')
        progress.tick(ok=True)
    records: list[dict] = []
    with ckpt.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    search_times = [float(r['xqdrant_indexed']['search_s']) for r in records if 'xqdrant_indexed' in r]
    metrics_indexed = {'mode': 'indexed_search_only', 'n_queries': len(records), 'note': 'Production path: query vector already in index / precomputed embedding', 'xqdrant': {**summarize_times(search_times), 'search': summarize_times(search_times), 'automated_steps': 1, 'manual_steps': 0}}
    if (results_dir / 'metrics.json').exists():
        cold = json.loads((results_dir / 'metrics.json').read_text())
        xq_cold = cold.get('xqdrant', {}).get('median_s')
        if xq_cold and metrics_indexed['xqdrant']['median_s']:
            metrics_indexed['speedup_cold_vs_indexed_median'] = round(float(xq_cold) / float(metrics_indexed['xqdrant']['median_s']), 2)
        fs_med = cold.get('foldseek_tmalign_pymol', {}).get('automated_only', {}).get('median_s')
        if fs_med and metrics_indexed['xqdrant']['median_s']:
            metrics_indexed['speedup_xq_indexed_vs_foldseek_auto_median'] = round(float(fs_med) / float(metrics_indexed['xqdrant']['median_s']), 2)
    write_json(results_dir / 'metrics_indexed.json', metrics_indexed)
    write_workflow_table(results_dir / 'workflow_comparison_indexed.tsv', [{'workflow': 'xqdrant_indexed_search', 'automated_steps': 1, 'manual_steps': 0, 'median_automated_s': metrics_indexed['xqdrant']['median_s'], 'median_total_s': metrics_indexed['xqdrant']['median_s'], 'p25_s': metrics_indexed['xqdrant']['p25_s'], 'p75_s': metrics_indexed['xqdrant']['p75_s'], 'rationale': 'automatic (pre-indexed embed + dims_explained)'}])
    summary_path = results_dir / 'summary.tsv'
    extra = [{'metric': 'xqdrant_indexed_median_search_s', 'value': str(metrics_indexed['xqdrant']['median_s'])}, {'metric': 'speedup_xq_indexed_vs_foldseek_auto', 'value': str(metrics_indexed.get('speedup_xq_indexed_vs_foldseek_auto_median'))}]
    if summary_path.exists():
        lines = summary_path.read_text().strip().splitlines()
        existing = {ln.split('\t')[0] for ln in lines[1:] if '\t' in ln}
        with summary_path.open('a') as f:
            for row in extra:
                if row['metric'] not in existing:
                    f.write(f"{row['metric']}\t{row['value']}\n")
    else:
        write_workflow_table(summary_path, extra)
    print(json.dumps(metrics_indexed, indent=2))
    return 0

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--restart', action='store_true')
    ap.add_argument('--max-queries', type=int, default=None)
    ap.add_argument('--skip-embed', action='store_true')
    ap.add_argument('--indexed', action='store_true', help='Search-only pass (pre-indexed embeddings) → metrics_indexed.json')
    ap.add_argument('--url', default=None)
    args = ap.parse_args()
    fcfg = load_yaml('exp_f.yaml')
    xcfg = load_yaml('xqdrant.yaml')
    corpus_cfg = load_yaml('corpus.yaml')
    esm_cfg = load_yaml('esm2.yaml')
    smoke = bool(args.smoke)
    corpus_dir = resolve_path(fcfg.get('corpus_smoke_dir' if smoke else 'corpus_dir', corpus_cfg['paper_outdir']))
    emb_dir = resolve_path(fcfg.get('embeddings_smoke_dir' if smoke else 'embeddings_dir'))
    base = resolve_path(fcfg.get('baselines_smoke_dir' if smoke else 'baselines_dir'))
    results_dir = ensure_dir(resolve_path(fcfg['smoke_results_dir'] if smoke else fcfg['results_dir']))
    ckpt = results_dir / 'per_query_timing.jsonl'
    run_config_path = results_dir / 'run_config.json'
    if args.indexed:
        if not run_config_path.exists():
            print(f'ERROR: need full Exp F run first ({run_config_path}).', file=sys.stderr)
            return 1
        run_cfg = json.loads(run_config_path.read_text())
        queries = list(run_cfg.get('query_ids') or [])
        if not queries:
            print('ERROR: run_config.json has no query_ids', file=sys.stderr)
            return 1
        emb_dir = resolve_path(run_cfg.get('embeddings', fcfg.get('embeddings_dir')))
        ids, emb, id_to_row = load_embeddings(emb_dir)
        url = args.url or xcfg.get('url', 'http://127.0.0.1:6333')
        coll = run_cfg.get('collection') or (xcfg['smoke_collection'] if smoke else xcfg['collection'])
        top_k = int(run_cfg.get('top_k', fcfg.get('top_k', 10)))
        dims_top = int(xcfg.get('dims_explained_top', 32))
        client = XQdrantREST(url)
        try:
            client.wait_ready(retries=10, sleep_s=0.5)
        except RuntimeError as e:
            print(f'ERROR: {e}. Start ./start_xqdrant.sh --daemon', file=sys.stderr)
            return 1
        return run_indexed_pass(results_dir=results_dir, emb_dir=emb_dir, id_to_row=id_to_row, emb=emb, queries=queries, client=client, coll=coll, top_k=top_k, dims_top=dims_top, q_filter=train_only_filter(), restart=bool(args.restart))
    if args.restart and results_dir.exists():
        for p in results_dir.glob('*'):
            if p.is_file():
                p.unlink()
        if ckpt.parent.exists():
            for p in ckpt.parent.glob('per_query_timing.jsonl'):
                p.unlink()
    struct_dir = base / 'structures'
    train_fa = base / 'fasta' / 'train.fasta'
    hold_fa = base / 'fasta' / 'holdout.fasta'
    if not struct_dir.exists() or not hold_fa.exists():
        print('ERROR: missing baseline inputs. Run ./run_exp_a_baselines.sh first.', file=sys.stderr)
        return 1
    foldseek = which_or_die('foldseek')
    blastp = which_optional('blastp')
    makeblastdb = which_optional('makeblastdb')
    if blastp is None or makeblastdb is None:
        print('WARN: BLAST+ not found — skipping blast_tmalign_pymol workflow. Install via ./fetch_tools_linux.sh (ncbi-blast+).', file=sys.stderr)
    tmalign = which_tmalign()
    meta = load_meta(corpus_dir / 'chains.csv')
    ids, emb, id_to_row = load_embeddings(emb_dir)
    fasta = load_fasta(hold_fa)
    n_queries = args.max_queries
    if n_queries is None:
        n_queries = int(fcfg.get('n_queries_smoke' if smoke else 'n_queries_paper', 50))
    seed = int(fcfg.get('random_seed', 42))
    queries = select_queries(meta, id_to_row=id_to_row, struct_dir=struct_dir, fasta=fasta, n=n_queries, seed=seed)
    if not queries:
        print('ERROR: no eligible holdout queries.', file=sys.stderr)
        return 1
    top_k = int(fcfg.get('top_k', 10))
    tm_cand = int(fcfg.get('tmalign_candidates', 10))
    pymol_s = float(fcfg.get('pymol_manual_minutes', 10)) * 60.0
    include_embed = bool(fcfg.get('include_embed_step', True)) and (not args.skip_embed)
    fscfg = fcfg.get('foldseek') or {}
    bcfg = fcfg.get('blast') or {}
    train_dir = base / 'structures_train'
    if not train_dir.exists() or not any(train_dir.glob('*.pdb')):
        ensure_dir(train_dir)
        for pdb in struct_dir.glob('*.pdb'):
            cid = pdb.stem
            if meta.get(cid, {}).get('split') == 'train':
                link = train_dir / pdb.name
                if not link.exists():
                    link.symlink_to(pdb.resolve())
    setup = measure_setup_costs(base=base, foldseek=foldseek, makeblastdb=makeblastdb, train_fa=train_fa, train_struct_dir=train_dir)
    write_json(results_dir / 'setup_costs.json', setup)
    target_db = base / 'foldseek_db' / 'train'
    blast_prefix = base / 'blast_db' / 'train'
    url = args.url or xcfg.get('url', 'http://127.0.0.1:6333')
    coll = xcfg['smoke_collection'] if smoke else xcfg['collection']
    dims_top = int(xcfg.get('dims_explained_top', 32))
    client = XQdrantREST(url)
    try:
        client.wait_ready(retries=10, sleep_s=0.5)
    except RuntimeError as e:
        print(f'ERROR: {e}. Start ./start_xqdrant.sh --daemon', file=sys.stderr)
        return 1
    if not client.collection_exists(coll):
        print(f'ERROR: collection {coll} missing. Run index_xqdrant.py', file=sys.stderr)
        return 1
    embedder: ESM2SingleEmbedder | None = None
    if include_embed:
        model_dir = resolve_path(esm_cfg.get('local_dir', 'models/esm2_t33_650M_UR50D'))
        if not model_dir.exists():
            print(f'ERROR: ESM2 model missing at {model_dir}', file=sys.stderr)
            return 1
        print('Loading ESM2 for per-query embed timing (once)...')
        embedder = ESM2SingleEmbedder(model_dir=model_dir, device=str(fcfg.get('embed_device', 'auto')), max_len=int(esm_cfg.get('max_seq_len', 1024)))
    done: set[str] = set()
    if ckpt.exists():
        with ckpt.open() as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)['query'])
    pending = [q for q in queries if q not in done]
    print(f'Exp F: {len(pending)} pending / {len(queries)} queries (embed={include_embed}, top_k={top_k}, tmalign≤{tm_cand})')
    run_config = {'mode': 'smoke' if smoke else 'paper', 'n_queries': len(queries), 'query_ids': queries, 'include_embed_step': include_embed, 'top_k': top_k, 'tmalign_candidates': tm_cand, 'pymol_manual_s': pymol_s, 'collection': coll, 'embeddings': str(emb_dir.relative_to(BIO_ROOT)), 'baselines': str(base.relative_to(BIO_ROOT))}
    write_json(results_dir / 'run_config.json', run_config)
    tmp_root = ensure_dir(results_dir / '_tmp')
    q_filter = train_only_filter()
    progress = BatchProgress(max(len(pending), 1), label='exp-f')
    for qid in pending:
        q_pdb = struct_dir / f'{qid}.pdb'
        seq = fasta[qid]
        rec: dict = {'query': qid, 'steps': {}}
        xq = {}
        if include_embed and embedder is not None:
            vec, embed_t = embedder.embed_sequence(seq)
            xq['embed_s'] = round(embed_t, 4)
        else:
            row = id_to_row[qid]
            vec = emb[row].astype(np.float64).tolist()
            xq['embed_s'] = 0.0
            xq['embed_note'] = 'precomputed'
        search_t, n_attr = time_xqdrant_query(client, collection=coll, vector=vec, limit=top_k, dims_top=dims_top, train_filter=q_filter)
        xq['search_s'] = round(search_t, 4)
        xq['n_hits_with_attribution'] = n_attr
        xq['total_automated_s'] = round(xq['embed_s'] + xq['search_s'], 4)
        xq['automated_steps'] = 2 if include_embed else 1
        xq['manual_steps'] = 0
        xq['delivers_rationale'] = 'automatic (dims_explained per hit)'
        rec['xqdrant'] = xq
        fs_tmp = ensure_dir(tmp_root / f'foldseek_{qid}')
        fs_t, fs_hits = time_foldseek_query(foldseek=foldseek, query_pdb=q_pdb, target_db=target_db, tmp_root=fs_tmp, sensitivity=float(fscfg.get('sensitivity', 9.5)), max_seqs=int(fscfg.get('max_seqs', 50)), threads=int(fscfg.get('threads', 1)))
        tm_t, tm_n = time_tmalign_candidates(tmalign=tmalign, query_pdb=q_pdb, struct_dir=struct_dir, candidates=fs_hits, max_candidates=tm_cand)
        fs_wf = {'foldseek_s': round(fs_t, 4), 'tmalign_s': round(tm_t, 4), 'tmalign_pairs': tm_n, 'pymol_manual_s': pymol_s, 'total_automated_s': round(fs_t + tm_t, 4), 'total_with_manual_s': round(fs_t + tm_t + pymol_s, 4), 'automated_steps': 2, 'manual_steps': 1, 'delivers_rationale': 'manual PyMOL after structural alignment'}
        rec['foldseek_tmalign_pymol'] = fs_wf
        blast_db_ready = (blast_prefix.parent / 'train.pin').exists() or (blast_prefix.parent / 'train.pdb').exists()
        if blastp and blast_db_ready:
            bl_tmp = ensure_dir(tmp_root / f'blast_{qid}')
            bl_t, bl_hits = time_blast_query(blastp=blastp, query_seq=seq, query_id=qid, db_prefix=blast_prefix, tmp_root=bl_tmp, evalue=float(bcfg.get('evalue', 10)), max_target_seqs=int(bcfg.get('max_target_seqs', 50)), num_threads=int(bcfg.get('num_threads', 1)))
            tm2_t, tm2_n = time_tmalign_candidates(tmalign=tmalign, query_pdb=q_pdb, struct_dir=struct_dir, candidates=bl_hits, max_candidates=tm_cand)
            bl_wf = {'blast_s': round(bl_t, 4), 'tmalign_s': round(tm2_t, 4), 'tmalign_pairs': tm2_n, 'pymol_manual_s': pymol_s, 'total_automated_s': round(bl_t + tm2_t, 4), 'total_with_manual_s': round(bl_t + tm2_t + pymol_s, 4), 'automated_steps': 2, 'manual_steps': 1, 'delivers_rationale': 'manual PyMOL after sequence+structure pipeline'}
            rec['blast_tmalign_pymol'] = bl_wf
        elif blastp:
            rec['blast_tmalign_pymol'] = {'skipped': 'blast_db_missing'}
        with ckpt.open('a') as f:
            f.write(json.dumps(rec, sort_keys=True) + '\n')
        progress.tick(ok=True)
    records: list[dict] = []
    with ckpt.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    def col(wf_key: str, step: str) -> list[float]:
        return [float(r[wf_key].get(step, 0.0)) for r in records if wf_key in r and r[wf_key].get(step) is not None]
    xq_auto = [float(r['xqdrant']['total_automated_s']) for r in records if 'xqdrant' in r]
    fs_auto = [float(r['foldseek_tmalign_pymol']['total_automated_s']) for r in records if 'foldseek_tmalign_pymol' in r]
    fs_full = [float(r['foldseek_tmalign_pymol']['total_with_manual_s']) for r in records if 'foldseek_tmalign_pymol' in r]
    bl_auto = [float(r['blast_tmalign_pymol']['total_automated_s']) for r in records if 'blast_tmalign_pymol' in r and 'total_automated_s' in r['blast_tmalign_pymol']]
    bl_full = [float(r['blast_tmalign_pymol']['total_with_manual_s']) for r in records if 'blast_tmalign_pymol' in r and 'total_with_manual_s' in r['blast_tmalign_pymol']]
    metrics = {'mode': 'smoke' if smoke else 'paper', 'n_queries': len(records), 'pymol_manual_s': pymol_s, 'xqdrant': {**summarize_times(xq_auto), 'embed': summarize_times(col('xqdrant', 'embed_s')), 'search': summarize_times(col('xqdrant', 'search_s')), 'automated_steps': 2 if include_embed else 1, 'manual_steps': 0}, 'foldseek_tmalign_pymol': {'automated_only': summarize_times(fs_auto), 'with_manual_pymol': summarize_times(fs_full), 'foldseek': summarize_times(col('foldseek_tmalign_pymol', 'foldseek_s')), 'tmalign': summarize_times(col('foldseek_tmalign_pymol', 'tmalign_s')), 'automated_steps': 2, 'manual_steps': 1}, 'blast_tmalign_pymol': {'skipped': len(bl_auto) == 0, 'automated_only': summarize_times(bl_auto), 'with_manual_pymol': summarize_times(bl_full), 'blast': summarize_times(col('blast_tmalign_pymol', 'blast_s')), 'tmalign': summarize_times(col('blast_tmalign_pymol', 'tmalign_s')), 'automated_steps': 2, 'manual_steps': 1}, 'speedup_xq_vs_foldseek_auto_median': round(float(np.median(fs_auto)) / float(np.median(xq_auto)), 2) if xq_auto and fs_auto and (np.median(xq_auto) > 0) else None, 'speedup_xq_vs_blast_auto_median': round(float(np.median(bl_auto)) / float(np.median(xq_auto)), 2) if xq_auto and bl_auto and (np.median(xq_auto) > 0) else None}
    write_json(results_dir / 'metrics.json', metrics)
    wf_rows = [{'workflow': 'xqdrant_attributed_search', 'automated_steps': metrics['xqdrant']['automated_steps'], 'manual_steps': metrics['xqdrant']['manual_steps'], 'median_automated_s': metrics['xqdrant']['median_s'], 'median_total_s': metrics['xqdrant']['median_s'], 'p25_s': metrics['xqdrant']['p25_s'], 'p75_s': metrics['xqdrant']['p75_s'], 'rationale': 'automatic (dims_explained)'}, {'workflow': 'foldseek_tmalign_pymol', 'automated_steps': 2, 'manual_steps': 1, 'median_automated_s': metrics['foldseek_tmalign_pymol']['automated_only']['median_s'], 'median_total_s': metrics['foldseek_tmalign_pymol']['with_manual_pymol']['median_s'], 'p25_s': metrics['foldseek_tmalign_pymol']['with_manual_pymol']['p25_s'], 'p75_s': metrics['foldseek_tmalign_pymol']['with_manual_pymol']['p75_s'], 'rationale': 'manual PyMOL inspection'}]
    if bl_auto:
        wf_rows.append({'workflow': 'blast_tmalign_pymol', 'automated_steps': 2, 'manual_steps': 1, 'median_automated_s': metrics['blast_tmalign_pymol']['automated_only']['median_s'], 'median_total_s': metrics['blast_tmalign_pymol']['with_manual_pymol']['median_s'], 'p25_s': metrics['blast_tmalign_pymol']['with_manual_pymol']['p25_s'], 'p75_s': metrics['blast_tmalign_pymol']['with_manual_pymol']['p75_s'], 'rationale': 'manual PyMOL inspection'})
    write_workflow_table(results_dir / 'workflow_comparison.tsv', wf_rows)
    summary_rows = [{'metric': 'n_queries', 'value': str(len(records))}, {'metric': 'xqdrant_median_automated_s', 'value': str(metrics['xqdrant']['median_s'])}, {'metric': 'foldseek_tmalign_median_automated_s', 'value': str(metrics['foldseek_tmalign_pymol']['automated_only']['median_s'])}, {'metric': 'foldseek_tmalign_median_with_pymol_s', 'value': str(metrics['foldseek_tmalign_pymol']['with_manual_pymol']['median_s'])}, {'metric': 'blast_tmalign_median_automated_s', 'value': str(metrics['blast_tmalign_pymol']['automated_only']['median_s'])}, {'metric': 'speedup_xq_vs_foldseek_auto', 'value': str(metrics['speedup_xq_vs_foldseek_auto_median'])}, {'metric': 'speedup_xq_vs_blast_auto', 'value': str(metrics['speedup_xq_vs_blast_auto_median'])}, {'metric': 'pymol_manual_estimate_s', 'value': str(pymol_s)}]
    write_workflow_table(results_dir / 'summary.tsv', summary_rows)
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    print(json.dumps(metrics, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
