"""Experiment B v3 — improved attribution validity (does not overwrite v1/v2).

Changes vs v2 (exp_b_v2_attribution.py):
  - More pairs: top-10 Exp A neighbors (was 5)
  - Larger dim→profile map: all train chains with binned DSSP profiles (was 3k subset)
  - Optional ridge map (config map_method)
  - Vectorized fast row permutations (5000 default)
  - Reframed hypothesis tests:
      * global_row_perm — v2-style replication
      * same_fold_row_perm — primary: homolog pairs only
      * label_perm_fold_gap — fold-label shuffle null for specificity
  - Fixed ablations: re-score pairs per top-N (v2 reused cached cohort means)
  - Attribution concentration diagnostics (top-1 mass, entropy)

Artifacts under results/exp_b_v3[/ _smoke]/ only.
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
from numpy.linalg import solve
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import BIO_ROOT, BatchProgress, ensure_dir, load_yaml, resolve_path, write_json
from exp_b_v2_attribution import append_jsonl, cohort_label, contrib_vector, fit_dim_feature_map, global_feature_cosine, iter_exp_a, load_embeddings, load_features_table, load_jsonl, load_profiles, load_split, load_v1_metrics, mann_whitney_auroc, pair_scores, pearson, profile_agreement, spearman, truncate_contrib, write_comparison_tsv, zscore_cols

def fit_map(emb: np.ndarray, feats: np.ndarray, method: str, ridge_alpha: float) -> np.ndarray:
    if method == 'ridge':
        ez = zscore_cols(emb)
        fz = zscore_cols(feats)
        d = ez.shape[1]
        return solve(ez.T @ ez + ridge_alpha * np.eye(d), ez.T @ fz)
    return fit_dim_feature_map(emb, feats)

def load_train_profiles_for_map(ann_dir: Path, splits: dict[str, str], id_to_row: dict[str, int], n_bins: int, n_profile_feat: int) -> dict[str, np.ndarray]:
    from annotate_dssp import binned_profile
    expected_len = n_bins * n_profile_feat
    out: dict[str, np.ndarray] = {}
    for p in ann_dir.glob('*.json'):
        cid = p.stem
        if splits.get(cid) != 'train' or cid not in id_to_row:
            continue
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        bp = data.get('binned_profile')
        if bp and bp.get('values'):
            vec = np.array(bp['values'], dtype=np.float64)
            if vec.size == expected_len:
                out[cid] = vec
                continue
        residues = data.get('residues')
        if residues:
            vec = np.array(binned_profile(residues, {}, n_bins), dtype=np.float64)
            if vec.size == expected_len:
                out[cid] = vec
    return out

def rank_rows(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, axis=1), axis=1).astype(np.float64)

def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    den = np.where(den < 1e-12, 1.0, den)
    return (a * b).sum(axis=1) / den

def spearman_rows(pred: np.ndarray, y: np.ndarray, y_ranks: np.ndarray | None=None) -> np.ndarray:
    ry = y_ranks if y_ranks is not None else rank_rows(y)
    return pearson_rows(rank_rows(pred), ry)

def build_contrib_matrix(contrib: dict[str, float], dim: int, top_n: int | None=None) -> np.ndarray:
    return contrib_vector(contrib, dim, top_n)

def attribution_diagnostics(a: np.ndarray) -> dict[str, float]:
    top1 = float(a.max(axis=1).mean())
    ent = []
    for row in a:
        pos = row[row > 1e-12]
        if pos.size == 0:
            ent.append(0.0)
        else:
            ent.append(float(-(pos * np.log(pos)).sum()))
    return {'mean_top1_mass': top1, 'mean_entropy': float(np.mean(ent)), 'frac_top1_above_0_9': float(np.mean(a.max(axis=1) >= 0.9))}

def row_permutation_test(a: np.ndarray, y: np.ndarray, m: np.ndarray, n_perm: int, seed: int, mask: np.ndarray | None=None) -> tuple[float, float, float]:
    y_ranks = rank_rows(y)
    scores = spearman_rows(a @ m, y, y_ranks)
    if mask is not None:
        scores = scores[mask]
    real = float(scores.mean()) if scores.size else 0.0
    prng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        mp = m[prng.permutation(m.shape[0])]
        s = spearman_rows(a @ mp, y, y_ranks)
        if mask is not None:
            s = s[mask]
        null[i] = float(s.mean()) if s.size else 0.0
        if (i + 1) % 500 == 0 or i + 1 == n_perm:
            print(f'  [row_perm] {i + 1}/{n_perm}', flush=True)
    p = float((np.sum(null >= real) + 1) / (n_perm + 1))
    return (real, p, float(null.mean()))

def label_permutation_gap(scores: np.ndarray, labels: np.ndarray, n_perm: int, seed: int) -> tuple[float, float, float]:
    real = float(scores[labels].mean() - scores[~labels].mean())
    prng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        lp = prng.permutation(labels)
        null[i] = float(scores[lp].mean() - scores[~lp].mean())
        if (i + 1) % 500 == 0 or i + 1 == n_perm:
            print(f'  [label_perm] {i + 1}/{n_perm}', flush=True)
    p = float((np.sum(null >= real) + 1) / (n_perm + 1))
    return (real, p, float(null.mean()))

def analyze_cohort_v3(records: list[dict], a_full: np.ndarray, y: np.ndarray, m: np.ndarray, labels: np.ndarray, cos_threshold: float, n_perm: int, seed: int) -> dict:
    idx = [i for i, r in enumerate(records) if float(r['feature_cosine']) <= cos_threshold + 1e-12]
    if not idx:
        return {'n_pairs': 0, 'cohort': cohort_label(cos_threshold)}
    sub_recs = [records[i] for i in idx]
    sub_a = a_full[idx]
    sub_y = y[idx]
    sub_labels = labels[idx]
    y_ranks = rank_rows(sub_y)
    spearmans = spearman_rows(sub_a @ m, sub_y, y_ranks)
    pearsons = pearson_rows(sub_a @ m, sub_y)
    real_all, p_global, null_global = row_permutation_test(sub_a, sub_y, m, n_perm, seed, mask=None)
    real_sf, p_sf, null_sf = row_permutation_test(sub_a, sub_y, m, n_perm, seed + 1, mask=sub_labels)
    gap, p_gap, null_gap = label_permutation_gap(spearmans, sub_labels, n_perm, seed + 2)
    auc, mw_p = mann_whitney_auroc(spearmans, sub_labels)
    same = spearmans[sub_labels]
    diff = spearmans[~sub_labels]
    return {'cohort': cohort_label(cos_threshold), 'feature_cosine_max': cos_threshold, 'n_pairs': len(sub_recs), 'mean_pearson': float(pearsons.mean()), 'median_pearson': float(np.median(pearsons)), 'mean_spearman': float(spearmans.mean()), 'median_spearman': float(np.median(spearmans)), 'p_value_global_row_perm': p_global, 'null_mean_global_row_perm': null_global, 'p_value_same_fold_row_perm': p_sf, 'null_mean_same_fold_row_perm': null_sf, 'real_mean_same_fold_row_perm': real_sf, 'p_value_label_perm_fold_gap': p_gap, 'null_mean_label_perm_fold_gap': null_gap, 'real_label_perm_fold_gap': gap, 'significant_0_05_global_row_perm': p_global < 0.05, 'significant_0_05_same_fold_row_perm': p_sf < 0.05, 'significant_0_05_label_perm_gap': p_gap < 0.05, 'mean_spearman_same_fold': float(same.mean()) if len(same) else None, 'mean_spearman_diff_fold': float(diff.mean()) if len(diff) else None, 'n_same_fold': int(sub_labels.sum()), 'n_diff_fold': int((~sub_labels).sum()), 'auroc_same_fold': auc, 'mann_whitney_p': mw_p, 'p_value_spearman': p_global, 'null_mean_spearman': null_global, 'significant_0_05_spearman': p_global < 0.05}

def ablation_stats(pair_payloads: list[tuple[dict, np.ndarray, np.ndarray]], m: np.ndarray, ranges: np.ndarray, top_n: int, n_perm: int, seed: int) -> dict:
    dim = m.shape[0]
    a = np.stack([build_contrib_matrix(d, dim, top_n) for d, _, _ in pair_payloads])
    y = np.stack([profile_agreement(pq, ph, ranges) for _, pq, ph in pair_payloads])
    y_ranks = rank_rows(y)
    scores = spearman_rows(a @ m, y, y_ranks)
    real = float(scores.mean())
    prng = np.random.default_rng(seed)
    null = np.empty(min(n_perm, 500), dtype=np.float64)
    for i in range(len(null)):
        mp = m[prng.permutation(m.shape[0])]
        null[i] = float(spearman_rows(a @ mp, y, y_ranks).mean())
    p = float((np.sum(null >= real) + 1) / (len(null) + 1))
    return {'mean_spearman': real, 'p_value_spearman': p, 'significant_0_05_spearman': p < 0.05, 'attribution': attribution_diagnostics(a)}

def write_v2_comparison(path: Path, v2: dict, v3_primary: dict) -> None:
    rows = [('metric', 'v2', 'v3'), ('n_pairs', v2.get('n_pairs'), v3_primary.get('n_pairs')), ('n_train_map', v2.get('n_train_map'), v3_primary.get('n_train_map')), ('mean_spearman', v2.get('primary_cohort', {}).get('mean_spearman'), v3_primary.get('mean_spearman')), ('p_global_row_perm', v2.get('primary_cohort', {}).get('p_value_spearman'), v3_primary.get('p_value_global_row_perm')), ('p_same_fold_row_perm', None, v3_primary.get('p_value_same_fold_row_perm')), ('p_label_perm_fold_gap', None, v3_primary.get('p_value_label_perm_fold_gap')), ('mean_spearman_same_fold', v2.get('primary_cohort', {}).get('mean_spearman_same_fold'), v3_primary.get('mean_spearman_same_fold')), ('mean_spearman_diff_fold', v2.get('primary_cohort', {}).get('mean_spearman_diff_fold'), v3_primary.get('mean_spearman_diff_fold')), ('mann_whitney_p', v2.get('primary_cohort', {}).get('mann_whitney_p'), v3_primary.get('mann_whitney_p'))]
    with path.open('w') as f:
        for a, b, c in rows:
            f.write(f'{a}\t{b}\t{c}\n')

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--restart', action='store_true')
    ap.add_argument('--max-pairs', type=int, default=None)
    ap.add_argument('--n-perm', type=int, default=None)
    ap.add_argument('--top-dims', type=int, default=None)
    args = ap.parse_args()
    bcfg = load_yaml('exp_b_v3.yaml')
    corpus_cfg = load_yaml('corpus.yaml')
    smoke = bool(args.smoke)
    results_dir = resolve_path(bcfg['smoke_results_dir'] if smoke else bcfg['results_dir'])
    if args.restart and results_dir.exists():
        print(f'--restart: clearing {results_dir}')
        shutil.rmtree(results_dir)
    ensure_dir(results_dir)
    ckpt_dir = ensure_dir(results_dir / 'checkpoints')
    pairs_jsonl = ckpt_dir / 'pairs.jsonl'
    global_names = list(bcfg['global_features'])
    feat_table = resolve_path(bcfg['features_table'])
    ann_dir = resolve_path(bcfg['annotations_dir'])
    n_bins = int(bcfg.get('n_bins', 32))
    n_profile_feat = len(bcfg.get('profile_features') or [])
    profile_dim = n_bins * n_profile_feat
    map_method = str(bcfg.get('map_method', 'corr'))
    ridge_alpha = float(bcfg.get('ridge_alpha', 1.0))
    if not feat_table.exists():
        print(f'ERROR: missing {feat_table}', file=sys.stderr)
        return 1
    emb_dir = resolve_path('embeddings/smoke' if smoke else 'embeddings/esm2_t33_650M')
    corpus_dir = resolve_path(corpus_cfg['smoke_outdir'] if smoke else corpus_cfg['paper_outdir'])
    exp_a_jsonl = resolve_path(bcfg['exp_a_smoke_jsonl'] if smoke else bcfg['exp_a_jsonl'])
    exp_a_legacy = exp_a_jsonl.parent.parent / 'per_query.json'
    if not exp_a_jsonl.exists() and (not exp_a_legacy.exists()):
        print(f'ERROR: missing Exp A results ({exp_a_jsonl})', file=sys.stderr)
        return 1
    ids, emb, id_to_row = load_embeddings(emb_dir)
    global_map = load_features_table(feat_table, global_names)
    splits = load_split(corpus_dir / 'chains.csv')
    max_neigh = int(bcfg.get('max_neighbors_per_query', 10))
    min_mass = float(bcfg.get('min_attribution_mass', 1e-06))
    needed: set[str] = set()
    for rec in iter_exp_a(exp_a_jsonl, exp_a_legacy):
        needed.add(rec['query'])
        for n in (rec.get('neighbors') or [])[:max_neigh]:
            needed.add(n['chain_id'])
    profiles_map = load_profiles(ann_dir, needed, n_bins, n_profile_feat)
    if len(profiles_map) < 30:
        print(f'ERROR: only {len(profiles_map)} chains with binned profiles. Run: annotate_dssp.py --from-exp-a --store-profiles', file=sys.stderr)
        return 1
    map_profiles = load_train_profiles_for_map(ann_dir, splits, id_to_row, n_bins, n_profile_feat)
    train_ids = sorted(map_profiles.keys())
    n_map_cfg = int(bcfg.get('n_train_for_map', 0))
    if smoke:
        n_map_cfg = min(n_map_cfg or 80, len(train_ids), 80)
    if n_map_cfg > 0 and len(train_ids) > n_map_cfg:
        rng = np.random.default_rng(int(bcfg.get('map_random_seed', 42)))
        pick = rng.choice(len(train_ids), size=n_map_cfg, replace=False)
        train_ids = [train_ids[i] for i in sorted(pick)]
    if len(train_ids) < 50:
        print(f'ERROR: only {len(train_ids)} train chains for map', file=sys.stderr)
        return 1
    e = np.stack([emb[id_to_row[c]] for c in train_ids], axis=0)
    f = np.stack([map_profiles[c] for c in train_ids], axis=0)
    m = fit_map(e, f, map_method, ridge_alpha)
    ranges = f.max(axis=0) - f.min(axis=0)
    map_path = results_dir / 'dim_profile_map.npz'
    np.savez_compressed(map_path, M=m, n_bins=n_bins, map_method=map_method, train_ids=np.array(train_ids), feature_ranges=ranges)
    write_json(results_dir / 'dim_profile_map.json', {'n_train': len(train_ids), 'map_method': map_method, 'ridge_alpha': ridge_alpha if map_method == 'ridge' else None, 'dim': int(m.shape[0]), 'profile_dim': profile_dim, 'max_abs_weight': float(np.abs(m).max()), 'mean_abs_weight': float(np.abs(m).mean())})
    top_n = args.top_dims
    planned: list[tuple[str, str, dict]] = []
    for rec in iter_exp_a(exp_a_jsonl, exp_a_legacy):
        q = rec['query']
        if q not in profiles_map or q not in id_to_row or q not in global_map:
            continue
        for n in (rec.get('neighbors') or [])[:max_neigh]:
            hid = n['chain_id']
            dims = n.get('dims_explained') or {}
            if hid not in profiles_map or hid not in id_to_row or hid not in global_map:
                continue
            if not dims:
                continue
            if sum((abs(float(v)) for v in dims.values())) < min_mass:
                continue
            planned.append((q, hid, dims))
    if args.max_pairs is not None:
        planned = planned[:args.max_pairs]
    done = {(r['query'], r['hit']) for r in load_jsonl(pairs_jsonl)}
    pending = [(q, h, d) for q, h, d in planned if (q, h) not in done]
    print(f'Exp B v3 | map={map_method} train={len(train_ids)} neighbors={max_neigh} planned={len(planned)} pending={len(pending)}')
    t0 = time.time()
    progress = BatchProgress(max(len(pending), 1), label='expB-v3-pairs')
    if not pending:
        progress.tick(skipped=True, force_print=True)
    for i, (q, h, dims) in enumerate(pending):
        pq, ph = (profiles_map[q], profiles_map[h])
        gq, gh = (global_map[q], global_map[h])
        ps, ss = pair_scores(dims, m, pq, ph, ranges, top_n)
        append_jsonl(pairs_jsonl, {'query': q, 'hit': h, 'pearson': ps, 'spearman': ss, 'feature_cosine': global_feature_cosine(gq, gh), 'n_dims': len(truncate_contrib(dims, top_n)), 'attribution_mass': sum((abs(float(v)) for v in truncate_contrib(dims, top_n).values())), 'unix': int(time.time())})
        progress.tick(ok=True, force_print=i + 1 >= len(pending))

    def fold_of(cid: str) -> str:
        p = ann_dir / f'{cid}.json'
        if p.exists():
            return json.loads(p.read_text()).get('scop_fold') or ''
        return ''
    records = load_jsonl(pairs_jsonl)
    wanted = {(q, h) for q, h, _ in planned}
    records = [r for r in records if (r['query'], r['hit']) in wanted]
    for r in records:
        fq, fh = (fold_of(r['query']), fold_of(r['hit']))
        r['same_fold'] = bool(fq and fq == fh)
        r['scop_fold_query'] = fq
        r['scop_fold_hit'] = fh
    dims_lookup: dict[tuple[str, str], dict] = {}
    for rec in iter_exp_a(exp_a_jsonl, exp_a_legacy):
        q = rec['query']
        for n in (rec.get('neighbors') or [])[:max_neigh]:
            dims_lookup[q, n['chain_id']] = n.get('dims_explained') or {}
    pair_payloads = [(dims_lookup.get((r['query'], r['hit']), {}), profiles_map[r['query']], profiles_map[r['hit']]) for r in records]
    dim = m.shape[0]
    a_full = np.stack([build_contrib_matrix(d, dim, top_n) for d, _, _ in pair_payloads])
    y = np.stack([profile_agreement(pq, ph, ranges) for _, pq, ph in pair_payloads])
    labels = np.array([bool(r.get('same_fold')) for r in records])
    attr_diag = attribution_diagnostics(a_full)
    n_perm = int(args.n_perm or bcfg.get('n_permutations', 5000))
    seed = int(bcfg.get('permutation_seed', 42))
    cohort_thresholds = [float(x) for x in bcfg.get('feature_cosine_cohorts') or [1.0]]
    print(f'Running hypothesis tests ({n_perm} permutations)...')
    cohorts = [analyze_cohort_v3(records, a_full, y, m, labels, t, n_perm, seed) for t in cohort_thresholds]
    primary = next((c for c in cohorts if c.get('cohort') == 'all'), cohorts[0])
    sat_min = float(bcfg.get('saturated_diff_fold_cosine_min', 0.999))
    sat_recs = [r for r in records if not r.get('same_fold') and float(r['feature_cosine']) >= sat_min]
    sat_scores = [float(r['spearman']) for r in sat_recs]
    same_scores = [float(r['spearman']) for r in records if r.get('same_fold')]
    ablations = {}
    for n_top in bcfg.get('ablation_top_dims') or []:
        if top_n is not None and n_top != top_n:
            continue
        ablations[f'top_{n_top}'] = ablation_stats(pair_payloads, m, ranges, int(n_top), min(n_perm, 500), seed + int(n_top))
    v1_path = resolve_path(bcfg.get('exp_b_v1_metrics', 'results/exp_b/metrics.json'))
    v2_path = resolve_path(bcfg.get('exp_b_v2_metrics', 'results/exp_b_v2/metrics.json'))
    v1 = load_v1_metrics(v1_path)
    v2 = load_v1_metrics(v2_path)
    metrics = {'version': 'exp_b_v3', 'mode': 'smoke' if smoke else 'paper', 'n_pairs': len(records), 'n_train_map': len(train_ids), 'map_method': map_method, 'max_neighbors_per_query': max_neigh, 'profile_dim': profile_dim, 'n_bins': n_bins, 'top_dims_ablation': top_n, 'attribution_diagnostics': attr_diag, 'primary_cohort': primary, 'primary_hypothesis': 'same_fold_row_perm', 'cohorts': cohorts, 'ablations': ablations, 'negative_control': {'saturated_diff_fold_cosine_min': sat_min, 'n_saturated_diff_fold': len(sat_recs), 'mean_spearman_saturated_diff_fold': float(np.mean(sat_scores)) if sat_scores else None, 'mean_spearman_same_fold': float(np.mean(same_scores)) if same_scores else None}, 'v1_reference': {'path': str(v1_path.relative_to(BIO_ROOT)) if v1_path.exists() else None, 'mean_pair_score': v1.get('mean_pair_score'), 'p_value_right_tail': v1.get('p_value_right_tail')}, 'v2_reference': {'path': str(v2_path.relative_to(BIO_ROOT)) if v2_path.exists() else None, 'mean_spearman': v2.get('primary_cohort', {}).get('mean_spearman'), 'p_value_global_row_perm': v2.get('primary_cohort', {}).get('p_value_spearman'), 'n_pairs': v2.get('n_pairs'), 'n_train_map': v2.get('n_train_map')}, 'n_permutations': n_perm, 'elapsed_s': round(time.time() - t0, 1), 'completed_unix': int(time.time())}
    write_json(results_dir / 'metrics.json', metrics)
    write_json(results_dir / 'pairs.json', {'pairs': records})
    with (results_dir / 'summary.tsv').open('w') as f:
        f.write('section\tmetric\tvalue\n')
        for k, v in primary.items():
            f.write(f'primary\t{k}\t{v}\n')
        for c in cohorts:
            label = c.get('cohort', '?')
            for k, v in c.items():
                if k != 'cohort':
                    f.write(f'cohort.{label}\t{k}\t{v}\n')
        for name, st in ablations.items():
            for k, v in st.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        f.write(f'ablation.{name}\t{sk}\t{sv}\n')
                else:
                    f.write(f'ablation.{name}\t{k}\t{v}\n')
        for k, v in metrics['attribution_diagnostics'].items():
            f.write(f'attribution\t{k}\t{v}\n')
        nc = metrics['negative_control']
        for k, v in nc.items():
            f.write(f'negative_control\t{k}\t{v}\n')
    write_v2_comparison(results_dir / 'v2_vs_v3_comparison.tsv', v2, primary)
    if v1_path.exists():
        write_comparison_tsv(results_dir / 'v1_vs_v3_comparison.tsv', v1, primary)
    print(json.dumps({'primary': {'same_fold_row_perm_p': primary.get('p_value_same_fold_row_perm'), 'label_perm_gap_p': primary.get('p_value_label_perm_fold_gap'), 'global_row_perm_p': primary.get('p_value_global_row_perm')}, 'v2_reference': metrics['v2_reference']}, indent=2))
    print(f'Wrote {results_dir.relative_to(BIO_ROOT)}/')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
