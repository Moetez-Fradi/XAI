"""Shared helpers for Experiment E — functional site mapping and localization."""
from __future__ import annotations
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
import numpy as np
from annotate_dssp import KD
from exp_b_v2_attribution import contrib_vector, fit_dim_feature_map, load_embeddings, load_split
from exp_b_v3_attribution import load_train_profiles_for_map
_SITE_TYPES_DEFAULT = {'active site', 'binding site', 'site', 'metal binding'}

def normalize_feature_type(t: str) -> str:
    return re.sub('\\s+', ' ', (t or '').strip().lower())

def load_sifts_chain_uniprot(csv_path: Path) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    with csv_path.open(newline='') as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            if header is None:
                header = [c.strip() for c in row]
                continue
            rec = {header[i]: row[i].strip() for i in range(len(header))}
            pdb = (rec.get('PDB') or rec.get('pdb') or '').lower()
            chain = (rec.get('CHAIN') or rec.get('chain') or '').strip()
            if not pdb or not chain:
                continue
            key = (pdb, chain)
            out.setdefault(key, []).append(rec)
    return out

def load_sifts_chain_table(csv_path: Path, key_cols: tuple[str, ...], val_col: str) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = {}
    with csv_path.open(newline='') as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            if header is None:
                header = [c.strip() for c in row]
                continue
            rec = {header[i]: row[i].strip() for i in range(len(header))}
            pdb = (rec.get('PDB') or rec.get('pdb') or '').lower()
            chain = (rec.get('CHAIN') or rec.get('chain') or '').strip()
            val = rec.get(val_col, '').strip()
            if pdb and chain and val:
                out.setdefault((pdb, chain), set()).add(val)
    return out

def uniprot_pos_to_pdb_resseq(uniprot_pos: int, segments: list[dict]) -> int | None:
    for seg in segments:
        try:
            sp_beg = int(seg.get('SP_BEG') or seg.get('RES_BEG') or 0)
            sp_end = int(seg.get('SP_END') or seg.get('RES_END') or 0)
            pdb_beg = int(seg.get('PDB_BEG') or 0)
        except (TypeError, ValueError):
            continue
        if sp_beg <= uniprot_pos <= sp_end:
            return pdb_beg + (uniprot_pos - sp_beg)
    return None

def parse_uniprot_feature_positions(entry: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feat in entry.get('features') or []:
        ftype = normalize_feature_type(str(feat.get('type') or ''))
        desc = str(feat.get('description') or '')
        loc = feat.get('location') or {}
        start = loc.get('start') or {}
        end = loc.get('end') or {}
        s = start.get('value') if isinstance(start, dict) else start
        e = end.get('value') if isinstance(end, dict) else end
        if s is None:
            continue
        if e is None:
            e = s
        try:
            s_i, e_i = (int(s), int(e))
        except (TypeError, ValueError):
            continue
        rows.append({'type': ftype, 'description': desc, 'start': s_i, 'end': e_i})
    return rows

def specific_go_overlap(q_go: set[str], n_go: set[str], generic: set[str]) -> bool:
    q = {g for g in q_go if g and g not in generic}
    n = {g for g in n_go if g and g not in generic}
    return bool(q & n)

def neighbor_pair_label(q_ec: set[str], q_go: set[str], q_fold: str, q_uniprot: str, n_ec: set[str], n_go: set[str], n_fold: str, n_uniprot: str, *, ec_level: int, generic_go: set[str]) -> str:
    if q_uniprot and n_uniprot and (q_uniprot == n_uniprot):
        return 'functional_match'
    if q_ec and n_ec and shared_ec(q_ec, n_ec, ec_level):
        return 'functional_match'
    if specific_go_overlap(q_go, n_go, generic_go):
        return 'functional_match'
    if fold_key(q_fold) == fold_key(n_fold) and fold_key(q_fold):
        return 'fold_only'
    return 'unrelated'

def fold_key(sccs: str) -> str:
    bits = (sccs or '').split('.')
    return '.'.join(bits[:2]) if len(bits) >= 2 else sccs

def ensure_benchmark_dssp(chain_ids: set[str], ann_dir: Path, mkdssp_cfg: str, corpus_csv: Path) -> None:
    from common_bio import resolve_path, write_json
    from exp_d_thermo import ensure_dssp
    ensure_dssp(chain_ids, ann_dir, mkdssp_cfg, corpus_csv)

def ec_at_level(ec: str, level: int) -> str:
    parts = [p for p in (ec or '').strip().split('.') if p]
    if not parts:
        return ''
    return '.'.join(parts[:max(1, min(level, len(parts)))])

def shared_ec(ec_a: set[str], ec_b: set[str], level: int) -> bool:
    a = {ec_at_level(x, level) for x in ec_a if x}
    b = {ec_at_level(x, level) for x in ec_b if x}
    a.discard('')
    b.discard('')
    return bool(a & b)

def load_uniprot_entry(json_path: Path) -> dict | None:
    if not json_path.exists() or json_path.stat().st_size == 0:
        return None
    try:
        data = json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return None
    if isinstance(data, list) and data:
        return data[0]
    return data if isinstance(data, dict) else None

def map_sites_for_chain(chain_id: str, uniprot_acc: str, uniprot_entry: dict, sifts_segments: list[dict], residues: list[dict], *, site_types: set[str], catalytic_keywords: list[str], buffer: int) -> dict[str, Any]:
    pdb_id, chain = chain_id.split('_', 1)
    kw = [k.lower() for k in catalytic_keywords]
    site_resseq: set[int] = set()
    mapped_features: list[dict[str, Any]] = []
    resseq_to_idx = {int(r['resseq']): i for i, r in enumerate(residues) if r.get('resseq') is not None}
    idx_to_resseq = {i: int(r['resseq']) for i, r in enumerate(residues) if r.get('resseq') is not None}
    for feat in parse_uniprot_feature_positions(uniprot_entry):
        ftype = feat['type']
        desc = feat.get('description') or ''
        desc_l = desc.lower()
        is_site = ftype in site_types
        is_catalytic = any((k in desc_l for k in kw))
        if not (is_site or (ftype == 'site' and is_catalytic)):
            continue
        for upos in range(int(feat['start']), int(feat['end']) + 1):
            rs = uniprot_pos_to_pdb_resseq(upos, sifts_segments)
            if rs is None:
                continue
            site_resseq.add(rs)
            mapped_features.append({'uniprot_pos': upos, 'resseq': rs, 'type': ftype, 'description': desc})
    if buffer > 0 and site_resseq:
        expanded: set[int] = set()
        for rs in site_resseq:
            if rs not in resseq_to_idx:
                expanded.add(rs)
                continue
            i0 = resseq_to_idx[rs]
            for j in range(max(0, i0 - buffer), min(len(residues), i0 + buffer + 1)):
                if j in idx_to_resseq:
                    expanded.add(idx_to_resseq[j])
        site_resseq = expanded
    return {'chain_id': chain_id, 'uniprot_accession': uniprot_acc, 'pdb_id': pdb_id.lower(), 'chain': chain, 'n_residues': len(residues), 'site_resseq': sorted(site_resseq), 'n_site_residues': len(site_resseq), 'features': mapped_features}

def load_dssp_residues(ann_dir: Path, chain_id: str) -> list[dict]:
    p = ann_dir / f'{chain_id}.json'
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return data.get('residues') or []

def residue_feature_vector(r: dict, bin_idx: int, n_bins: int, n_feat: int=5) -> np.ndarray:
    out = np.zeros(n_bins * n_feat, dtype=np.float64)
    base = bin_idx * n_feat
    ss = (r.get('ss') or '-')[0]
    if ss in {'H', 'G', 'I'}:
        out[base] = 1.0
    elif ss in {'E', 'B'}:
        out[base + 1] = 1.0
    else:
        out[base + 2] = 1.0
    aa = r.get('aa', 'X')
    out[base + 3] = (KD.get(aa, 0.0) + 4.5) / 9.0
    out[base + 4] = float(r.get('rsa') or 0.0)
    return out

def residue_attribution_scores(residues: list[dict], contrib: dict[str, float], M: np.ndarray, n_bins: int=32) -> np.ndarray:
    a = contrib_vector(contrib, M.shape[0], None)
    if a.sum() < 1e-12:
        return np.zeros(len(residues), dtype=np.float64)
    n = max(len(residues), 1)
    scores = np.zeros(n, dtype=np.float64)
    for i, r in enumerate(residues):
        bi = min(int(i * n_bins / n), n_bins - 1)
        fv = residue_feature_vector(r, bi, n_bins)
        scores[i] = float(a @ M @ fv)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores

def site_enrichment(scores: np.ndarray, site_mask: np.ndarray) -> dict[str, float]:
    site_mask = site_mask.astype(bool)
    n = len(scores)
    if n == 0 or site_mask.sum() == 0 or (~site_mask).sum() == 0:
        return {'site_mean': float('nan'), 'background_mean': float('nan'), 'enrichment_ratio': float('nan'), 'site_fraction': float('nan')}
    site_mean = float(np.mean(scores[site_mask]))
    bg_mean = float(np.mean(scores[~site_mask]))
    ratio = site_mean / bg_mean if bg_mean > 1e-12 else float('nan')
    return {'site_mean': site_mean, 'background_mean': bg_mean, 'enrichment_ratio': ratio, 'site_fraction': float(site_mask.sum() / n)}

def site_auc(scores: np.ndarray, site_mask: np.ndarray) -> float:
    labels = site_mask.astype(bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float('nan')
    order = np.argsort(-scores)
    y = labels[order]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    tpr = np.cumsum(y) / n_pos
    fpr = np.cumsum(~y) / n_neg
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))

def resseq_site_mask(residues: list[dict], site_resseq: set[int] | list[int]) -> np.ndarray:
    site = set((int(x) for x in site_resseq))
    return np.array([int(r.get('resseq') or -1) in site for r in residues], dtype=bool)

def which_tmalign() -> str:
    from common_bio import BIO_ROOT
    candidates = [BIO_ROOT / 'tools/mamba/envs/bio-tools/bin/TMalign', BIO_ROOT / 'tools/mamba/envs/bio-tools/bin/tmalign']
    for p in candidates:
        if p.exists() and os.access(p, os.X_OK):
            return str(p.resolve())
    for name in ('TMalign', 'tmalign'):
        p = shutil.which(name)
        if p:
            return p
    raise FileNotFoundError('TMalign not found. Install via bio-tools conda env or ./fetch_tools_linux.sh')

def parse_tmalign_stdout(stdout: str) -> list[tuple[int, int]]:
    lines = stdout.splitlines()
    start = None
    for i, line in enumerate(lines):
        if 'denotes residue pairs' in line:
            start = i + 1
            break
    if start is None or start + 2 >= len(lines):
        return []
    s1, mid, s2 = (lines[start], lines[start + 1], lines[start + 2])
    if len({len(s1), len(mid), len(s2)}) != 1:
        return []
    mapping: list[tuple[int, int]] = []
    i = j = 0
    for c1, cm, c2 in zip(s1, mid, s2):
        if cm == ' ':
            continue
        if c1 != '-':
            i += 1
        if c2 != '-':
            j += 1
        if c1 != '-' and c2 != '-':
            mapping.append((i, j))
    return mapping

def tmalign_mapping(bin_path: str, q_pdb: Path, t_pdb: Path) -> list[tuple[int, int]]:
    if not q_pdb.exists() or not t_pdb.exists():
        return []
    proc = subprocess.run([bin_path, str(q_pdb), str(t_pdb)], capture_output=True, text=True, check=False)
    if proc.returncode != 0 and (not proc.stdout):
        return []
    return parse_tmalign_stdout(proc.stdout or '')

def tmalign_site_overlap(aligned_query_indices: set[int], site_seq_indices: set[int], n_res: int) -> dict[str, float]:
    n_aln = len(aligned_query_indices)
    n_site = len(site_seq_indices)
    if n_res <= 0 or n_aln == 0 or n_site == 0:
        return {'aligned_in_site_fraction': float('nan'), 'expected_fraction': float('nan'), 'enrichment_ratio': float('nan'), 'n_aligned': float(n_aln)}
    overlap = len(aligned_query_indices & site_seq_indices)
    frac = overlap / n_aln
    expected = n_site / n_res
    ratio = frac / expected if expected > 1e-12 else float('nan')
    return {'aligned_in_site_fraction': frac, 'expected_fraction': expected, 'enrichment_ratio': ratio, 'n_aligned': float(n_aln), 'n_overlap': float(overlap)}

def site_seq_indices_from_resseq(residues: list[dict], site_resseq: set[int] | list[int]) -> set[int]:
    site = set((int(x) for x in site_resseq))
    return {i for i, r in enumerate(residues) if int(r.get('resseq') or -1) in site}

def load_dim_map(ecfg: dict, ann_dir: Path, emb_dir: Path, splits: dict, id_to_row: dict) -> tuple[np.ndarray, np.ndarray]:
    p = Path(ecfg.get('exp_b_v3_map', 'results/exp_b_v3/dim_profile_map.npz'))
    if not p.is_absolute():
        from common_bio import resolve_path
        p = resolve_path(p)
    if p.exists():
        z = np.load(p)
        return (z['M'], z['feature_ranges'])
    profiles = load_train_profiles_for_map(ann_dir, splits, id_to_row, int(ecfg.get('n_bins', 32)), 5)
    train_ids = sorted(profiles.keys())
    ids, emb, _ = load_embeddings(emb_dir)
    e = np.stack([emb[id_to_row[c]] for c in train_ids])
    f = np.stack([profiles[c] for c in train_ids])
    m = fit_dim_feature_map(e, f)
    return (m, f.max(0) - f.min(0))

def permutation_enrichment_null(scores: np.ndarray, site_mask: np.ndarray, n_perm: int, seed: int) -> tuple[float, float]:
    real = site_enrichment(scores, site_mask)['enrichment_ratio']
    if not np.isfinite(real):
        return (float('nan'), float('nan'))
    prng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    n = len(scores)
    k = int(site_mask.sum())
    for i in range(n_perm):
        mask = np.zeros(n, dtype=bool)
        mask[prng.choice(n, size=k, replace=False)] = True
        null[i] = site_enrichment(scores, mask)['enrichment_ratio']
    p = float((np.sum(null >= real) + 1) / (n_perm + 1))
    return (float(real), p)
