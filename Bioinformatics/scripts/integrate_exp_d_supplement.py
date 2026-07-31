"""Merge paper corpus + Exp D supplement, embed new chains, resume XQdrant index.

Does NOT rewrite corpus_lock.json (paper holdout stays immutable).
Supplement chains are always split=train.

Steps:
  1) Merge chains.csv → data/processed/corpus_merged/
  2) Append ESM2 embeddings for new chain IDs
  3) Resume index_xqdrant upsert from checkpoint (or full if needed)

Usage:
  ./integrate_exp_d_supplement.py
  ./integrate_exp_d_supplement.py --skip-index
"""
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import BIO_ROOT, ensure_dir, load_yaml, resolve_path, write_json
FIELDNAMES = ['chain_id', 'pdb_id', 'chain', 'length', 'split', 'scop_domain', 'scop_sccs', 'scop_fold', 'scop_class', 'uniprot', 'structure_file', 'sequence']

def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))

def merge_corpora(main_csv: Path, supp_csv: Path, out_dir: Path) -> tuple[int, int]:
    ensure_dir(out_dir)
    main_rows = load_rows(main_csv) if main_csv.exists() else []
    supp_rows = load_rows(supp_csv) if supp_csv.exists() else []
    merged: dict[str, dict] = {r['chain_id']: r for r in main_rows}
    n_new = 0
    for r in supp_rows:
        cid = r['chain_id']
        if cid in merged:
            continue
        merged[cid] = r
        n_new += 1
    rows = sorted(merged.values(), key=lambda r: r['chain_id'])
    out_csv = out_dir / 'chains.csv'
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDNAMES})
    train = [r['chain_id'] for r in rows if r.get('split') == 'train']
    holdout = [r['chain_id'] for r in rows if r.get('split') == 'holdout']
    (out_dir / 'split_train.txt').write_text('\n'.join(train) + '\n')
    (out_dir / 'split_holdout.txt').write_text('\n'.join(holdout) + '\n')
    write_json(out_dir / 'merge_manifest.json', {'main_chains': len(main_rows), 'supplement_chains': len(supp_rows), 'merged_chains': len(rows), 'new_from_supplement': n_new, 'n_train': len(train), 'n_holdout': len(holdout), 'note': 'Holdout IDs preserved from main corpus; supplement adds train only.'})
    return (len(rows), n_new)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--skip-embed', action='store_true')
    ap.add_argument('--skip-index', action='store_true')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    dcfg = load_yaml('exp_d.yaml')
    corpus_cfg = load_yaml('corpus.yaml')
    py = BIO_ROOT / '.venv' / 'bin' / 'python'
    main_dir = resolve_path(corpus_cfg['paper_outdir'])
    supp_dir = resolve_path(dcfg.get('corpus_supplement_dir', 'data/processed/corpus_supplement'))
    merged_dir = resolve_path(dcfg.get('corpus_merged_dir', 'data/processed/corpus_merged'))
    emb_dir = resolve_path(dcfg.get('embeddings_dir', 'embeddings/esm2_t33_650M'))
    supp_csv = supp_dir / 'chains.csv'
    if not supp_csv.exists():
        print(f'ERROR: missing {supp_csv}. Run build_exp_d_supplement.py first.', file=sys.stderr)
        return 1
    n_merged, n_new = merge_corpora(main_dir / 'chains.csv', supp_csv, merged_dir)
    print(f'Merged corpus: {n_merged} chains ({n_new} new from supplement) → {merged_dir}')
    if not args.skip_embed:
        cmd = [str(py), str(BIO_ROOT / 'scripts' / 'embed_esm2.py'), '--append', '--corpus-dir', str(merged_dir), '--outdir', str(emb_dir)]
        if args.device:
            cmd.extend(['--device', args.device])
        print('==> Append embeddings for new chains')
        subprocess.run(cmd, check=True, cwd=BIO_ROOT)
    if not args.skip_index:
        cmd = [str(py), str(BIO_ROOT / 'scripts' / 'index_xqdrant.py'), '--corpus-dir', str(merged_dir)]
        print('==> Sync XQdrant index (requires ./start_xqdrant.sh --daemon)')
        subprocess.run(cmd, check=True, cwd=BIO_ROOT)
    print('Integration complete. Re-run Exp D:')
    print('  ./run_exp_d.sh --restart --skip-oma')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
