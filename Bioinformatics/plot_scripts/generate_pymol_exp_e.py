"""PyMOL session for Exp E — attribution-colored query with functional sites highlighted.

Picks the functional-match pair with highest XQ site enrichment.

Outputs under plot_scripts/outputs/pymol/:
  exp_e_<query>_attribution.csv
  exp_e_<query>_site_mask.csv
  exp_e_session.pml
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BIO_ROOT, PYMOL_OUT, RESULTS, ensure_out
from exp_e_lib import load_dim_map, load_dssp_residues, load_embeddings, load_split, residue_attribution_scores
from common_bio import load_yaml, resolve_path
from exp_b_v2_attribution import iter_exp_a

def load_tsv(path: Path) -> list[dict]:
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split('\t')
    return [{header[i]: parts[i] if i < len(parts) else '' for i in range(len(header))} for parts in (ln.split('\t') for ln in lines[1:])]

def find_structure(pdb_id: str) -> str | None:
    pdb_dir = BIO_ROOT / 'data' / 'raw' / 'pdb'
    for name in (f'{pdb_id.upper()}.cif.gz', f'{pdb_id.upper()}.cif'):
        p = pdb_dir / name
        if p.exists():
            return str(p.relative_to(BIO_ROOT))
    return None

def main() -> int:
    ensure_out()
    loc_path = RESULTS / 'exp_e' / 'localization_comparison.tsv'
    proc_dir = BIO_ROOT / 'data' / 'processed' / 'exp_e'
    if not loc_path.exists():
        print('ERROR: run Exp E first', file=sys.stderr)
        return 1
    rows = [r for r in load_tsv(loc_path) if r.get('pair_label') == 'functional_match']
    if not rows:
        print('ERROR: no functional-match pairs', file=sys.stderr)
        return 1
    best = max(rows, key=lambda r: float(r.get('xq_enrichment_ratio') or 0))
    q, n = (best['query_chain_id'], best['neighbor_chain_id'])
    pdb_id, chain = q.split('_', 1)
    site_path = proc_dir / 'sites' / f'{q}.json'
    site_resseq = set(json.loads(site_path.read_text()).get('site_resseq') or [])
    ecfg = load_yaml('exp_e.yaml')
    ann_dir = resolve_path(ecfg['annotations_dir'])
    emb_dir = resolve_path(ecfg['embeddings_dir'])
    corpus_dir = resolve_path(ecfg['corpus_dir'])
    ids, emb, id_to_row = load_embeddings(emb_dir)
    splits = load_split(corpus_dir / 'chains.csv')
    M, _ = load_dim_map(ecfg, ann_dir, emb_dir, splits, id_to_row)
    exp_a = resolve_path(ecfg['exp_a_jsonl'])
    contrib = {}
    for rec in iter_exp_a(exp_a, exp_a.parent.parent / 'per_query.json'):
        qid = rec.get('query') or rec.get('query_chain_id') or rec.get('chain_id')
        if qid != q:
            continue
        for nb in rec.get('neighbors') or []:
            if nb.get('chain_id') == n:
                contrib = nb.get('dims_explained') or {}
                break
    residues = load_dssp_residues(ann_dir, q)
    scores = residue_attribution_scores(residues, contrib, M, int(ecfg.get('n_bins', 32)))
    attr_csv = PYMOL_OUT / f'exp_e_{q}_attribution.csv'
    site_csv = PYMOL_OUT / f'exp_e_{q}_site_mask.csv'
    with attr_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['resi', 'value'])
        for r, s in zip(residues, scores):
            w.writerow([r['resseq'], f'{s:.4f}'])
    with site_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['resi', 'value'])
        for r in residues:
            v = 1.0 if int(r['resseq']) in site_resseq else 0.0
            w.writerow([r['resseq'], f'{v:.1f}'])
    struct = find_structure(pdb_id)
    if not struct:
        print('ERROR: PDB not found', file=sys.stderr)
        return 1
    pml = PYMOL_OUT / 'exp_e_session.pml'
    rel = lambda p: str(p.relative_to(BIO_ROOT))
    lines = [f'# Exp E PyMOL — query {q} vs neighbor {n}', f"# xq_enrichment={best.get('xq_enrichment_ratio')}", 'reinitialize', 'bg_color white', f'load {struct}, prot', f'remove prot and not chain {chain}', 'create query_chain, prot', 'delete prot', 'python', 'import csv', 'from pymol import cmd', f"for csvpath, col in [(r'{rel(attr_csv)}', 'b'), (r'{rel(site_csv)}', 'q')]:", '    with open(csvpath) as f:', '        for row in csv.DictReader(f):', '            cmd.alter(f\'query_chain and resi {row["resi"]}\', f\'{col}={row["value"]}\')', "    cmd.sort('query_chain')", 'python end', 'spectrum b, yellow_red, query_chain, minimum=0, maximum=1', 'show cartoon, query_chain', 'color gray80, query_chain', 'show sticks, query_chain and q > 0.5', 'color tv_red, query_chain and q > 0.5', 'zoom query_chain, 8', '# Export: png plot_scripts/outputs/pymol/exp_e_active_site.png, 2400, 1800']
    pml.write_text('\n'.join(lines) + '\n')
    print(f'  → {pml.relative_to(BIO_ROOT)}')
    print(f"  Pair: {q} → {n} (xq_enrichment={best.get('xq_enrichment_ratio')})")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from _common import OUTDIR, save, setup_style
        setup_style()
        resseq = [int(r['resseq']) for r in residues]
        site_mask = [1.0 if s in site_resseq else 0.0 for s in resseq]
        fig, ax = plt.subplots(figsize=(10, 2.2))
        x = np.arange(len(scores))
        ax.fill_between(x, 0, scores, alpha=0.35, color='#2563eb', label='Attribution')
        ax.plot(x, scores, color='#2563eb', lw=1)
        for i, m in enumerate(site_mask):
            if m > 0.5:
                ax.axvline(i, color='#dc2626', alpha=0.25, lw=2)
        ax.set_xlabel('Residue index (DSSP order)')
        ax.set_ylabel('Attribution score')
        ax.set_title(f'Exp E - {q} per-residue attribution (UniProt sites in red)')
        ax.set_ylim(0, max(scores) * 1.15 + 1e-06)
        save(fig, f'exp_e_{q}_attribution')
        plt.close(fig)
        print(f'  → plot_scripts/outputs/exp_e_{q}_attribution.png')
    except Exception as e:
        print(f'WARN: matplotlib figure skipped: {e}', file=sys.stderr)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
