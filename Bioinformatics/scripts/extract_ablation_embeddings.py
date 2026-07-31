"""Extract ablation subset vectors from full-corpus embeddings (layer33_mean baseline)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import ensure_dir, load_yaml, resolve_path, write_json
from exp_b_attribution import load_embeddings

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default='embeddings/esm2_t33_650M')
    ap.add_argument('--outdir', default='embeddings/ablation/layer33_mean')
    args = ap.parse_args()
    acfg = load_yaml('exp_ablation.yaml')
    chains_file = resolve_path(acfg['processed_dir']) / 'chain_ids.txt'
    if not chains_file.exists():
        print('ERROR: run prep_exp_ablation.py first', file=sys.stderr)
        return 1
    want = [ln.strip() for ln in chains_file.read_text().splitlines() if ln.strip()]
    src_dir = resolve_path(args.source)
    outdir = ensure_dir(resolve_path(args.outdir))
    ids, emb, id_to_row = load_embeddings(src_dir)
    missing = [c for c in want if c not in id_to_row]
    if missing:
        print(f'ERROR: {len(missing)} chains missing from source embed', file=sys.stderr)
        return 1
    mat = np.stack([emb[id_to_row[c]] for c in want], axis=0).astype(np.float32)
    np.save(outdir / 'vectors.npy', mat)
    (outdir / 'ids.txt').write_text('\n'.join(want) + '\n')
    meta = {'source': str(src_dir), 'n': len(want), 'note': 'extracted subset (paper layer33 mean)'}
    write_json(outdir / 'meta.json', meta)
    print(f'Extracted {len(want)} vectors → {outdir}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
