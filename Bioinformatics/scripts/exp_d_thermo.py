"""Experiment D — thermophile/mesophile ortholog attribution case study.

For each meso→thermo pair:
  1) XQdrant retrieval: meso query searches train index; record thermo rank + dims_explained
  2) DSSP + structure thermostability proxies (charged surface, loops, ion pairs, packing/SASA)
  3) Attribution-weighted profile score (Exp B map) for the pair
  4) Aggregate: ortholog recovery, feature deltas, attribution vs delta correlation + significance

Artifacts under results/exp_d[/ _smoke]/
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import BIO_ROOT, BatchProgress, ensure_dir, load_yaml, resolve_path, write_json
from exp_b_v2_attribution import append_jsonl, contrib_vector, fit_dim_feature_map, load_embeddings, load_jsonl, load_profiles, load_split, pair_scores, pearson, spearman, zscore_cols
from exp_b_v3_attribution import load_train_profiles_for_map
from exp_d_stats import bootstrap_spearman_ci, perm_p_value_spearman
from exp_d_structure_features import ensure_structure_features, load_cached, merge_structure_features
from xqdrant_rest import XQdrantREST
THERMO_DELTA_KEYS = ['frac_charged_exposed', 'mean_loop_length', 'packing_density', 'ion_pair_density']
CHARGED = frozenset('DEKR')

def thermo_features(residues: list[dict]) -> dict:
    n = len(residues) or 1
    charged = charged_exp = buried = 0
    coil_runs: list[int] = []
    run = 0
    for r in residues:
        aa = r.get('aa', 'X')
        rsa = float(r.get('rsa') or 0.0)
        ss = (r.get('ss') or '-')[0]
        if aa in CHARGED:
            charged += 1
            if rsa > 0.25:
                charged_exp += 1
        if rsa < 0.25:
            buried += 1
        if ss in {' ', '-', 'T', 'S'}:
            run += 1
        else:
            if run:
                coil_runs.append(run)
            run = 0
    if run:
        coil_runs.append(run)
    return {'frac_charged': charged / n, 'frac_charged_exposed': charged_exp / n, 'frac_buried': buried / n, 'mean_loop_length': float(np.mean(coil_runs)) if coil_runs else 0.0, 'max_loop_length': float(max(coil_runs)) if coil_runs else 0.0, 'packing_proxy': 1.0 - float(np.mean([float(r.get('rsa') or 0) for r in residues])) if residues else 0.0}

def features_from_aggregates(agg: dict) -> dict:
    return {'frac_charged': None, 'frac_charged_exposed': None, 'frac_buried': 1.0 - float(agg.get('mean_rsa') or 0), 'mean_loop_length': float(agg.get('frac_coil') or 0) * float(agg.get('n_residues') or 300), 'max_loop_length': None, 'packing_proxy': 1.0 - float(agg.get('mean_rsa') or 0), 'frac_helix': float(agg.get('frac_helix') or 0), 'frac_sheet': float(agg.get('frac_sheet') or 0), 'frac_coil': float(agg.get('frac_coil') or 0), 'mean_rsa': float(agg.get('mean_rsa') or 0)}

def chain_features(ann_dir: Path, chain_id: str) -> dict:
    p = ann_dir / f'{chain_id}.json'
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    residues = data.get('residues') or []
    if residues:
        out = thermo_features(residues)
        agg = data.get('features') or data.get('aggregates') or {}
        out['frac_helix'] = float(agg.get('frac_helix') or 0)
        out['frac_coil'] = float(agg.get('frac_coil') or 0)
        out['mean_rsa'] = float(agg.get('mean_rsa') or 0)
        return out
    agg = data.get('features') or data.get('aggregates') or {}
    if agg:
        return features_from_aggregates(agg)
    return {}

def ensure_dssp(chain_ids: set[str], ann_dir: Path, mkdssp_cfg: str, corpus_csv: Path) -> None:
    from annotate_dssp import aggregate, chain_b_factors, find_mkdssp, find_structure, load_chain_targets, materialize_structure, parse_dssp_file, run_dssp_biopython
    missing = [c for c in chain_ids if not (ann_dir / f'{c}.json').exists()]
    need_residues = []
    for c in chain_ids:
        p = ann_dir / f'{c}.json'
        if p.exists():
            d = json.loads(p.read_text())
            if not d.get('residues'):
                need_residues.append(c)
    todo = sorted(set(missing) | set(need_residues))
    if not todo:
        return
    print(f'  DSSP annotate {len(todo)} chains (with residues)...')
    mk = find_mkdssp(mkdssp_cfg)
    pdb_dir = resolve_path('data/raw/pdb')
    import tempfile
    for cid in todo:
        targets = [t for t in load_chain_targets(corpus_csv, None, {cid})]
        if not targets:
            continue
        t = targets[0]
        src = find_structure(pdb_dir, t['pdb_id'])
        if not src:
            continue
        try:
            with tempfile.TemporaryDirectory(prefix='dssp_d_') as td:
                struct_path = materialize_structure(src, Path(td))
                residues = run_dssp_biopython(struct_path, mk, t['chain'])
                if not residues:
                    dssp_out = Path(td) / 'out.dssp'
                    subprocess.run([mk, str(struct_path), str(dssp_out)], check=True, capture_output=True, text=True)
                    residues = parse_dssp_file(dssp_out, t['chain'])
                bfac = chain_b_factors(struct_path, t['chain'])
                agg = aggregate(residues, bfac)
                payload = {'chain_id': cid, 'pdb_id': t['pdb_id'], 'chain': t['chain'], 'features': agg, 'residues': residues}
                write_json(ann_dir / f'{cid}.json', payload)
        except Exception as e:
            print(f'  WARN DSSP {cid}: {e}', file=sys.stderr)

def load_map(dcfg: dict, emb_dir: Path, ann_dir: Path, splits: dict, id_to_row: dict) -> tuple[np.ndarray, np.ndarray]:
    p = resolve_path(dcfg.get('exp_b_v3_map', 'results/exp_b_v3/dim_profile_map.npz'))
    if p.exists():
        z = np.load(p)
        return (z['M'], z['feature_ranges'])
    profiles = load_train_profiles_for_map(ann_dir, splits, id_to_row, 32, 5)
    train_ids = sorted(profiles.keys())
    ids, emb, _ = load_embeddings(emb_dir)
    e = np.stack([emb[id_to_row[c]] for c in train_ids])
    f = np.stack([profiles[c] for c in train_ids])
    m = fit_dim_feature_map(e, f)
    return (m, f.max(0) - f.min(0))

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--restart', action='store_true')
    args = ap.parse_args()
    dcfg = load_yaml('exp_d.yaml')
    xcfg = load_yaml('xqdrant.yaml')
    corpus_cfg = load_yaml('corpus.yaml')
    smoke = bool(args.smoke)
    from prep_exp_d_pairs import resolve_corpus_dir
    corpus_dir = resolve_corpus_dir(dcfg, corpus_cfg, smoke)
    results_dir = resolve_path(dcfg['smoke_results_dir'] if smoke else dcfg['results_dir'])
    if args.restart and results_dir.exists():
        shutil.rmtree(results_dir)
    ensure_dir(results_dir)
    ckpt = ensure_dir(results_dir / 'checkpoints') / 'pairs.jsonl'
    pairs_csv = resolve_path(dcfg['pairs_out'])
    if not pairs_csv.exists():
        print('ERROR: run prep_exp_d_pairs.py first', file=sys.stderr)
        return 1
    pairs: list[dict] = []
    with pairs_csv.open() as f:
        pairs = list(csv.DictReader(f))
    ann_dir = ensure_dir(resolve_path(dcfg['annotations_dir']))
    emb_dir = resolve_path(dcfg.get('embeddings_dir', 'embeddings/esm2_t33_650M'))
    ids, emb, id_to_row = load_embeddings(emb_dir)
    splits = load_split(corpus_dir / 'chains.csv')
    chain_ids = {p['meso_chain_id'] for p in pairs} | {p['thermo_chain_id'] for p in pairs}
    struct_dir = ensure_dir(resolve_path(dcfg.get('structure_features_dir', 'data/annotations/exp_d')))
    ion_cutoff = float(dcfg.get('ion_pair_cutoff_angstrom', 4.0))
    ensure_dssp(chain_ids, ann_dir, resolve_path(dcfg.get('mkdssp_bin', 'tools/mamba/envs/bio-tools/bin/mkdssp')), corpus_dir / 'chains.csv')
    ensure_structure_features(chain_ids, corpus_dir / 'chains.csv', struct_dir, ion_cutoff=ion_cutoff)
    m, ranges = load_map(dcfg, emb_dir, ann_dir, splits, id_to_row)
    profiles = load_profiles(ann_dir, chain_ids, 32, 5)
    url = str(xcfg.get('url', 'http://127.0.0.1:6333'))
    coll = str(xcfg.get('smoke_collection' if smoke else 'collection', 'proteins_esm2_650m'))
    top_k = int(dcfg.get('top_k', 20))
    search_split = str(dcfg.get('search_split', 'train'))
    dims_top = int(xcfg.get('dims_explained_top', 32))
    client = XQdrantREST(url)
    try:
        client.wait_ready(retries=5, sleep_s=0.5)
    except RuntimeError as e:
        print(f'ERROR: {e}\nStart ./start_xqdrant.sh', file=sys.stderr)
        return 1
    if not client.collection_exists(coll):
        print(f'ERROR: collection {coll} missing', file=sys.stderr)
        return 1
    qfilter = {'must': [{'key': 'split', 'match': {'value': search_split}}]}
    done = {(r['meso_chain_id'], r['thermo_chain_id']) for r in load_jsonl(ckpt)}
    pending = [p for p in pairs if (p['meso_chain_id'], p['thermo_chain_id']) not in done]
    print(f'Exp D | pairs={len(pairs)} pending={len(pending)} collection={coll}')
    t0 = time.time()
    progress = BatchProgress(max(len(pending), 1), label='expD-pairs')
    if not pending:
        progress.tick(skipped=True, force_print=True)
    for i, p in enumerate(pending):
        meso, thermo = (p['meso_chain_id'], p['thermo_chain_id'])
        rec: dict = {'meso_chain_id': meso, 'thermo_chain_id': thermo, 'source': p.get('source'), 'note': p.get('note')}
        if meso not in id_to_row:
            rec['error'] = 'meso_not_embedded'
            append_jsonl(ckpt, rec)
            progress.tick(fail=True)
            continue
        qvec = emb[id_to_row[meso]].astype(np.float32).tolist()
        hits = client.query(coll, qvec, limit=top_k, with_payload=True, with_dims_explained={'top': dims_top}, query_filter=qfilter)
        rank = None
        dims = None
        score = None
        for ri, h in enumerate(hits):
            pl = h.get('payload') or {}
            cid = str(pl.get('chain_id', h.get('id')))
            if cid == thermo:
                rank = ri
                score = float(h.get('score') or 0)
                dims = h.get('dims_explained')
                break
        rec['thermo_rank'] = rank
        rec['thermo_in_topk'] = rank is not None
        rec['retrieval_score'] = score
        if dims is None and thermo in id_to_row:
            direct = client.query(coll, qvec, limit=1, with_payload=True, with_dims_explained={'top': dims_top}, query_filter={'must': [{'key': 'chain_id', 'match': {'value': thermo}}]})
            if direct:
                h0 = direct[0]
                pl0 = h0.get('payload') or {}
                if str(pl0.get('chain_id', h0.get('id'))) == thermo:
                    dims = h0.get('dims_explained')
                    if score is None:
                        score = float(h0.get('score') or 0)
                    rec['retrieval_score'] = score
                    rec['direct_pair_query'] = True
        fm = merge_structure_features(chain_features(ann_dir, meso), load_cached(struct_dir, meso))
        ft = merge_structure_features(chain_features(ann_dir, thermo), load_cached(struct_dir, thermo))
        rec['features_meso'] = fm
        rec['features_thermo'] = ft
        delta = {k: float(ft[k]) - float(fm[k]) for k in fm if fm.get(k) is not None and ft.get(k) is not None}
        rec['features_delta'] = delta
        if dims and meso in profiles and (thermo in profiles):
            _, ss = pair_scores(dims, m, profiles[meso], profiles[thermo], ranges)
            rec['xq_profile_spearman'] = ss
            a = contrib_vector(dims, m.shape[0], None)
            rec['attribution_top1_mass'] = float(a.max())
            rec['n_dims_explained'] = len(dims)
        else:
            rec['xq_profile_spearman'] = None
        rec['unix'] = int(time.time())
        append_jsonl(ckpt, rec)
        progress.tick(ok=True, force_print=i + 1 >= len(pending))
    records = load_jsonl(ckpt)
    n = len(records)
    recovered = sum((1 for r in records if r.get('thermo_in_topk')))
    scores = [float(r['xq_profile_spearman']) for r in records if r.get('xq_profile_spearman') is not None]
    delta_keys = list(THERMO_DELTA_KEYS)
    delta_means = {}
    for k in delta_keys:
        vals = [float(r['features_delta'][k]) for r in records if r.get('features_delta') and r['features_delta'].get(k) is not None]
        delta_means[k] = float(np.mean(vals)) if vals else None
    n_perm = int(dcfg.get('n_permutations', 5000))
    n_boot = int(dcfg.get('n_bootstrap', 2000))
    seed = int(dcfg.get('permutation_seed', 42))
    by_source: dict[str, list[dict]] = {}
    for r in records:
        by_source.setdefault(str(r.get('source') or 'unknown'), []).append(r)
    source_metrics = {}
    for src, rs in sorted(by_source.items()):
        src_scores = [float(x['xq_profile_spearman']) for x in rs if x.get('xq_profile_spearman') is not None]
        source_metrics[src] = {'n_pairs': len(rs), 'recovery_rate': sum((1 for x in rs if x.get('thermo_in_topk'))) / len(rs) if rs else 0.0, 'mean_xq_profile_spearman': float(np.mean(src_scores)) if src_scores else None}
    corr_pairs = [r for r in records if r.get('xq_profile_spearman') is not None and r.get('features_delta')]
    attr_vs_delta = {}
    attr_vs_delta_p = {}
    attr_vs_delta_ci = {}
    attr_vs_delta_n = {}
    for ki, k in enumerate(delta_keys):
        xs = np.array([float(r['xq_profile_spearman']) for r in corr_pairs if r['features_delta'].get(k) is not None], dtype=np.float64)
        ys = np.array([float(r['features_delta'][k]) for r in corr_pairs if r['features_delta'].get(k) is not None], dtype=np.float64)
        if len(xs) >= 3:
            attr_vs_delta[k] = spearman(xs, ys)
            attr_vs_delta_n[k] = int(len(xs))
            _, p_val, _ = perm_p_value_spearman(xs, ys, n_perm, seed + ki)
            _, ci_lo, ci_hi = bootstrap_spearman_ci(xs, ys, n_boot, seed + 1000 + ki)
            attr_vs_delta_p[k] = p_val
            attr_vs_delta_ci[k] = {'lo': ci_lo, 'hi': ci_hi}
            print(f'  attr vs Δ{k}: ρ={attr_vs_delta[k]:.3f} p={p_val:.4f} CI=[{ci_lo:.3f},{ci_hi:.3f}] n={len(xs)}')
    metrics = {'version': 'exp_d', 'mode': 'smoke' if smoke else 'paper', 'n_pairs': n, 'ortholog_recovery_topk': {'top_k': top_k, 'n_recovered': recovered, 'rate': recovered / n if n else 0.0}, 'mean_xq_profile_spearman': float(np.mean(scores)) if scores else None, 'mean_feature_delta': delta_means, 'by_source': source_metrics, 'attribution_vs_delta_spearman': attr_vs_delta, 'attribution_vs_delta_p_value': attr_vs_delta_p, 'attribution_vs_delta_ci95': attr_vs_delta_ci, 'attribution_vs_delta_n': attr_vs_delta_n, 'n_permutations': n_perm, 'n_bootstrap': n_boot, 'permutation_seed': seed, 'thermo_delta_keys': delta_keys, 'ion_pair_cutoff_angstrom': ion_cutoff, 'sasa_method': 'biopython_shrake_rupley', 'pairs': records, 'elapsed_s': round(time.time() - t0, 1), 'completed_unix': int(time.time())}
    write_json(results_dir / 'metrics.json', metrics)
    with (results_dir / 'summary.tsv').open('w') as f:
        f.write('metric\tvalue\n')
        f.write(f'n_pairs\t{n}\n')
        f.write(f"recovery_rate@{top_k}\t{metrics['ortholog_recovery_topk']['rate']}\n")
        f.write(f"mean_xq_profile_spearman\t{metrics['mean_xq_profile_spearman']}\n")
        for k, v in delta_means.items():
            f.write(f'mean_delta_{k}\t{v}\n')
        for src, sm in source_metrics.items():
            f.write(f"source_{src}_n\t{sm['n_pairs']}\n")
            f.write(f"source_{src}_mean_spearman\t{sm['mean_xq_profile_spearman']}\n")
        for k, v in attr_vs_delta.items():
            f.write(f'attr_vs_delta_{k}_spearman\t{v}\n')
            if k in attr_vs_delta_p:
                f.write(f'attr_vs_delta_{k}_p_value\t{attr_vs_delta_p[k]}\n')
            if k in attr_vs_delta_ci:
                f.write(f"attr_vs_delta_{k}_ci95_lo\t{attr_vs_delta_ci[k]['lo']}\n")
                f.write(f"attr_vs_delta_{k}_ci95_hi\t{attr_vs_delta_ci[k]['hi']}\n")
    with (results_dir / 'significance.tsv').open('w') as f:
        f.write('feature_delta\tn\tspearman_rho\tp_value\tci95_lo\tci95_hi\tsignificant_0_05\n')
        for k in delta_keys:
            if k not in attr_vs_delta:
                continue
            ci = attr_vs_delta_ci.get(k, {})
            p = attr_vs_delta_p.get(k)
            sig = p is not None and p < 0.05
            f.write(f"{k}\t{attr_vs_delta_n.get(k, '')}\t{attr_vs_delta[k]}\t{p}\t{ci.get('lo', '')}\t{ci.get('hi', '')}\t{sig}\n")
    with (results_dir / 'pairs_by_source.tsv').open('w') as f:
        f.write('source\tn\trecovery_rate\tmean_xq_spearman\n')
        for src, sm in source_metrics.items():
            f.write(f"{src}\t{sm['n_pairs']}\t{sm['recovery_rate']}\t{sm['mean_xq_profile_spearman']}\n")
    with (results_dir / 'pymol_pairs.tsv').open('w') as f:
        f.write('meso_chain\tthermo_chain\tthermo_rank\txq_spearman\tdelta_charged_exp\tdelta_packing\tdelta_ion_pairs\tdelta_loop_len\n')
        for r in records:
            fd = r.get('features_delta') or {}
            f.write(f"{r['meso_chain_id']}\t{r['thermo_chain_id']}\t{r.get('thermo_rank')}\t{r.get('xq_profile_spearman')}\t{fd.get('frac_charged_exposed')}\t{fd.get('packing_density')}\t{fd.get('ion_pair_density')}\t{fd.get('mean_loop_length')}\n")
    print(json.dumps({'n_pairs': n, 'recovery_rate': metrics['ortholog_recovery_topk']['rate']}, indent=2))
    print(f'Wrote {results_dir.relative_to(BIO_ROOT)}/')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
