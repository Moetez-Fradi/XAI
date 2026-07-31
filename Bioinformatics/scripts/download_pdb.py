"""Download paper-scale PDB structures stratified by SCOPe fold.

Requires download_scop.py first.

Writes:
  data/raw/pdb/<PDBID>.cif.gz   (default; or .cif / .pdb)
  data/raw/pdb/paper_ids.txt
  data/raw/pdb/pilot_ids.txt    (same list when --pilot; kept for compat)
  data/raw/pdb/download_manifest.json
"""
from __future__ import annotations
import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import BIO_ROOT, BatchProgress, download_cfg, download_file, raw_dir, write_json

def _find_cla(scop_dir: Path) -> Path:
    matches = sorted(scop_dir.glob('dir.cla.scope*.txt'))
    if not matches:
        raise FileNotFoundError(f'No SCOPe class file in {scop_dir}. Run download_scop.py first.')
    return matches[0]

def _parse_pdb_and_fold(parts: list[str]) -> tuple[str | None, str | None]:
    pdb_id = None
    for tok in parts[1:4]:
        t = tok.strip().lower()
        if len(t) == 4 and t[0].isdigit() and t.isalnum():
            pdb_id = t
            break
    if pdb_id is None:
        sid = parts[0].lower()
        if len(sid) >= 5 and sid[0] == 'd' and sid[1].isdigit():
            pdb_id = sid[1:5]
    fold = None
    for tok in parts:
        if tok.count('.') >= 1 and tok[0].isalpha() and (tok[1] == '.'):
            bits = tok.split('.')
            if len(bits) >= 2 and bits[1].isdigit():
                fold = f'{bits[0]}.{bits[1]}'
                break
    return (pdb_id, fold)

def sample_pdb_ids_stratified(cla_path: Path, limit: int, *, stratify: bool, seed: int) -> list[str]:
    by_fold: dict[str, list[str]] = defaultdict(list)
    seen_global: set[str] = set()
    with cla_path.open() as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pdb_id, fold = _parse_pdb_and_fold(parts)
            if not pdb_id or pdb_id in seen_global:
                continue
            seen_global.add(pdb_id)
            by_fold[fold or 'unknown'].append(pdb_id)
    rng = random.Random(seed)
    for fold in by_fold:
        rng.shuffle(by_fold[fold])
    if not stratify or limit <= 0:
        flat = [pid for ids in by_fold.values() for pid in ids]
        rng.shuffle(flat)
        return flat if limit <= 0 else flat[:limit]
    folds = sorted(by_fold.keys())
    rng.shuffle(folds)
    pointers = {f: 0 for f in folds}
    selected: list[str] = []
    while len(selected) < limit:
        progressed = False
        for fold in folds:
            idx = pointers[fold]
            ids = by_fold[fold]
            if idx < len(ids):
                selected.append(ids[idx])
                pointers[fold] = idx + 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected

def _ext_and_url(base: str, pdb_id: str, fmt: str) -> tuple[str, str]:
    pid = pdb_id.upper()
    if fmt == 'cif.gz':
        return (f'{pid}.cif.gz', f'{base}/{pid}.cif.gz')
    if fmt == 'cif':
        return (f'{pid}.cif', f'{base}/{pid}.cif')
    return (f'{pid}.pdb', f'{base}/{pid}.pdb')

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--pilot', action='store_true', help='Small smoke set (pdb.pilot_unique_pdbs) instead of paper scale')
    ap.add_argument('--limit', type=int, default=None, help='Override unique PDB count')
    ap.add_argument('--ids-file', type=Path, default=None)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()
    cfg = download_cfg()['pdb']
    out = raw_dir('pdb')
    scop_dir = raw_dir('scop')
    mode = 'pilot' if args.pilot else 'paper'
    if args.ids_file:
        ids = [ln.strip().lower() for ln in args.ids_file.read_text().splitlines() if ln.strip() and (not ln.startswith('#'))]
    else:
        if args.limit is not None:
            limit = args.limit
        elif args.pilot:
            limit = int(cfg['pilot_unique_pdbs'])
        else:
            limit = int(cfg['paper_unique_pdbs'])
        cla = _find_cla(scop_dir)
        seed = args.seed if args.seed is not None else int(cfg.get('random_seed', 42))
        stratify = bool(cfg.get('stratify_by_fold', True))
        print(f'Sampling {limit} unique PDB IDs from {cla.name} (mode={mode}, stratify={stratify}, seed={seed})')
        ids = sample_pdb_ids_stratified(cla, limit, stratify=stratify, seed=seed)
    if not ids:
        print('No PDB IDs resolved.', file=sys.stderr)
        return 1
    paper_ids = out / 'paper_ids.txt'
    pilot_ids = out / 'pilot_ids.txt'
    paper_ids.write_text('\n'.join(ids) + '\n')
    pilot_ids.write_text('\n'.join(ids) + '\n')
    print(f'Will download {len(ids)} structures → {out}')
    fmt = cfg.get('format', 'cif.gz')
    base = cfg['base_url'].rstrip('/')
    sleep_s = float(cfg.get('sleep_s', 0.03))
    retries = int(cfg.get('max_retries', 4))
    ok, fail = ([], [])
    progress = BatchProgress(len(ids), label=f'PDB-{mode}')
    for i, pdb_id in enumerate(ids, 1):
        fname, url = _ext_and_url(base, pdb_id, fmt)
        dest = out / fname
        existed = dest.exists() and dest.stat().st_size > 0 and (not args.force)
        try:
            before = dest.stat().st_size if dest.exists() else 0
            download_file(url, dest, force=args.force, max_retries=retries, timeout=180, quiet=True)
            nbytes = dest.stat().st_size if dest.exists() else 0
            if existed:
                progress.tick(skipped=True, nbytes=0, force_print=i == len(ids))
            else:
                progress.tick(ok=True, nbytes=max(nbytes - before, nbytes), force_print=i == len(ids))
            ok.append(pdb_id)
        except Exception as e:
            recovered = False
            if fmt == 'cif.gz':
                try:
                    fname2, url2 = _ext_and_url(base, pdb_id, 'cif')
                    dest2 = out / fname2
                    download_file(url2, dest2, force=args.force, max_retries=retries, timeout=180, quiet=True)
                    ok.append(pdb_id)
                    progress.tick(ok=True, nbytes=dest2.stat().st_size if dest2.exists() else 0, force_print=i == len(ids))
                    recovered = True
                except Exception as e2:
                    e = e2
            if not recovered:
                print(f'  FAIL {pdb_id}: {e}', file=sys.stderr)
                fail.append({'id': pdb_id, 'error': str(e)})
                progress.tick(ok=False, force_print=i == len(ids))
        if sleep_s and i < len(ids):
            time.sleep(sleep_s)
    write_json(out / 'download_manifest.json', {'source': 'RCSB PDB', 'mode': mode, 'format': fmt, 'requested': len(ids), 'ok': len(ok), 'failed_count': len(fail), 'failed': fail[:50], 'ids_file': str(paper_ids.relative_to(BIO_ROOT)), 'outdir': str(out)})
    print(f'PDB {mode} done: {len(ok)} ok, {len(fail)} failed → {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
