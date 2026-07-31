"""Quick probe for Exp B v3 strategy selection (not part of paper pipeline)."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
from numpy.linalg import solve
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import resolve_path
from exp_b_v2_attribution import contrib_vector, fit_dim_feature_map, iter_exp_a, load_embeddings, load_profiles, load_split, profile_agreement, zscore_cols

def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    sa = np.sqrt((a * a).sum(1))
    sb = np.sqrt((b * b).sum(1))
    den = np.where(sa * sb < 1e-12, 1.0, sa * sb)
    return (a * b).sum(1) / den

def fit_ridge(emb: np.ndarray, feats: np.ndarray, alpha: float=1.0) -> np.ndarray:
    ez = zscore_cols(emb)
    fz = zscore_cols(feats)
    d = ez.shape[1]
    return solve(ez.T @ ez + alpha * np.eye(d), ez.T @ fz)

def load_fold_map(ann_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in ann_dir.glob('*.json'):
        data = json.loads(p.read_text())
        out[data['chain_id']] = data.get('scop_fold') or ''
    return out

def build_arrays(max_neighbors: int):
    emb_dir = resolve_path('embeddings/esm2_t33_650M')
    ann = resolve_path('data/annotations/dssp')
    exp_a = resolve_path('results/exp_a/checkpoints/queries.jsonl')
    exp_a_legacy = exp_a.parent.parent / 'per_query.json'
    ids, emb, id2 = load_embeddings(emb_dir)
    fold = load_fold_map(ann)
    needed: set[str] = set()
    for rec in iter_exp_a(exp_a, exp_a_legacy):
        needed.add(rec['query'])
        for nb in rec.get('neighbors') or []:
            needed.add(nb['chain_id'])
    profiles = load_profiles(ann, needed, 32, 5)
    splits = load_split(resolve_path('data/processed/corpus/chains.csv'))
    train_all = sorted((c for c, sp in splits.items() if sp == 'train' and c in profiles and (c in id2)))
    e = np.stack([emb[id2[c]] for c in train_all])
    f = np.stack([profiles[c] for c in train_all])
    ranges = f.max(0) - f.min(0)
    a_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    labels: list[bool] = []
    for rec in iter_exp_a(exp_a, exp_a_legacy):
        q = rec['query']
        if q not in profiles:
            continue
        for nb in (rec.get('neighbors') or [])[:max_neighbors]:
            h = nb['chain_id']
            dims = nb.get('dims_explained') or {}
            if h not in profiles or not dims:
                continue
            a_rows.append(contrib_vector(dims, 1280, None))
            y_rows.append(profile_agreement(profiles[q], profiles[h], ranges))
            labels.append(bool(fold.get(q)) and fold.get(q) == fold.get(h))
    return (np.stack(a_rows), np.stack(y_rows), np.array(labels), fit_dim_feature_map(e, f), fit_ridge(e, f, 1.0), len(train_all))

def perm_p(real: float, nulls: np.ndarray) -> float:
    return float((np.sum(nulls >= real) + 1) / (len(nulls) + 1))

def main() -> int:
    t0 = time.time()
    for max_n in (5, 10):
        a, y, labels, m_corr, m_ridge, n_train = build_arrays(max_n)
        for name, m in (('corr', m_corr), ('ridge1', m_ridge)):
            scores = pearson_rows(a @ m, y)
            gap = float(scores[labels].mean() - scores[~labels].mean())
            print(f'max_n={max_n} map={name} pairs={len(scores)} train={n_train} mean={scores.mean():.4f} sf={scores[labels].mean():.4f} df={scores[~labels].mean():.4f} gap={gap:.4f}')
    a, y, labels, m_corr, _, n_train = build_arrays(5)
    scores = pearson_rows(a @ m_corr, y)
    prng = np.random.default_rng(42)
    null_all = np.array([pearson_rows(a @ m_corr[prng.permutation(m_corr.shape[0])], y).mean() for _ in range(5000)])
    print(f'row_perm all: p={perm_p(float(scores.mean()), null_all):.4f} real={scores.mean():.4f} null={null_all.mean():.4f}')
    null_sf = np.array([pearson_rows(a @ m_corr[prng.permutation(m_corr.shape[0])], y)[labels].mean() for _ in range(5000)])
    print(f'row_perm same_fold: p={perm_p(float(scores[labels].mean()), null_sf):.4f} real={scores[labels].mean():.4f} null={null_sf.mean():.4f}')
    real_gap = float(scores[labels].mean() - scores[~labels].mean())
    null_gap = np.array([scores[prng.permutation(labels)].mean() - scores[~prng.permutation(labels)].mean() for _ in range(5000)])
    print(f'label_perm fold_gap: p={perm_p(real_gap, null_gap):.6f} gap={real_gap:.4f}')
    print(f'elapsed_s={time.time() - t0:.1f}')
    return 0

def spearman_perm_from_v2() -> None:
    from exp_b_v2_attribution import load_jsonl, profile_agreement
    results = resolve_path('results/exp_b_v2')
    z = np.load(results / 'dim_profile_map.npz')
    m, ranges = (z['M'], z['feature_ranges'])
    records = load_jsonl(results / 'checkpoints/pairs.jsonl')
    ann = resolve_path('data/annotations/dssp')
    exp_a = resolve_path('results/exp_a/checkpoints/queries.jsonl')
    exp_a_legacy = exp_a.parent.parent / 'per_query.json'

    def fold_of(cid: str) -> str:
        p = ann / f'{cid}.json'
        return json.loads(p.read_text()).get('scop_fold') or '' if p.exists() else ''
    labels = np.array([bool(fold_of(r['query']) and fold_of(r['query']) == fold_of(r['hit'])) for r in records])
    needed = {r['query'] for r in records} | {r['hit'] for r in records}
    profiles = load_profiles(ann, needed, 32, 5)
    dims_lookup: dict[tuple[str, str], dict] = {}
    for rec in iter_exp_a(exp_a, exp_a_legacy):
        q = rec['query']
        for n in (rec.get('neighbors') or [])[:5]:
            dims_lookup[q, n['chain_id']] = n.get('dims_explained') or {}
    a = np.stack([contrib_vector(dims_lookup[r['query'], r['hit']], 1280, None) for r in records])
    y = np.stack([profile_agreement(profiles[r['query']], profiles[r['hit']], ranges) for r in records])

    def rank_rows(x: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(x, axis=1), axis=1).astype(np.float64)
    ry = rank_rows(y)
    scores = pearson_rows(rank_rows(a @ m), ry)
    cached = np.array([float(r['spearman']) for r in records])
    print(f'v2 cached mean={cached.mean():.4f} recomputed={scores.mean():.4f}')
    prng = np.random.default_rng(42)
    real = float(scores.mean())
    nulls = np.array([pearson_rows(rank_rows(a @ m[prng.permutation(m.shape[0])]), ry).mean() for _ in range(5000)])
    print(f'spearman global row_perm p={perm_p(real, nulls):.4f} real={real:.4f} null={nulls.mean():.4f}')
    real_sf = float(scores[labels].mean())
    nulls_sf = np.array([pearson_rows(rank_rows(a @ m[prng.permutation(m.shape[0])]), ry)[labels].mean() for _ in range(5000)])
    print(f'spearman same_fold row_perm p={perm_p(real_sf, nulls_sf):.4f} real={real_sf:.4f} null={nulls_sf.mean():.4f}')
    real_gap = float(scores[labels].mean() - scores[~labels].mean())
    null_gap = np.array([scores[prng.permutation(labels)].mean() - scores[~prng.permutation(labels)].mean() for _ in range(5000)])
    print(f'spearman label_perm gap p={perm_p(real_gap, null_gap):.6f} gap={real_gap:.4f}')
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--v2-check':
        spearman_perm_from_v2()
    else:
        raise SystemExit(main())
