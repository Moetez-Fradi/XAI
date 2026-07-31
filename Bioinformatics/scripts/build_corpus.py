"""Filter PDB polymer chains, attach SCOPe/SIFTS labels, lock held-out split.

Writes (paper default → data/processed/corpus/):
  chains.csv
  sequences.fasta
  split_train.txt
  split_holdout.txt
  corpus_lock.json      # immutable holdout IDs + seed (refuses overwrite)
  build_manifest.json

Smoke mode (--smoke) writes to data/processed/corpus_smoke/ so it cannot
overwrite the paper lock.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import BIO_ROOT, BatchProgress, ensure_dir, load_yaml, raw_dir, resolve_path, write_json
AA = set('ACDEFGHIKLMNPQRSTVWY')
_CHAIN_RE = re.compile('([A-Za-z0-9])')

def _find_cla(scop_dir: Path) -> Path:
    matches = sorted(scop_dir.glob('dir.cla.scope*.txt'))
    if not matches:
        raise FileNotFoundError(f'No SCOPe class file in {scop_dir}')
    return matches[0]

def _parse_chain_field(field: str) -> list[str]:
    chains: list[str] = []
    for part in field.split(','):
        part = part.strip()
        if not part:
            continue
        head = part.split(':', 1)[0].strip()
        if not head:
            continue
        m = _CHAIN_RE.match(head)
        if m:
            chains.append(m.group(1).upper())
    return chains

def load_scop_chain_map(cla_path: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    with cla_path.open() as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            domain_id = parts[0]
            pdb_id = parts[1].lower()
            chain_field = parts[2]
            sccs = parts[3]
            if sccs.count('.') < 1:
                continue
            bits = sccs.split('.')
            fold = f'{bits[0]}.{bits[1]}' if len(bits) >= 2 else sccs
            family = sccs
            for ch in _parse_chain_field(chain_field):
                key = (pdb_id, ch)
                if key not in out:
                    out[key] = {'scop_domain': domain_id, 'scop_sccs': family, 'scop_fold': fold, 'scop_class': bits[0]}
    return out

def load_sifts_uniprot(sifts_csv: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    opener = gzip.open if sifts_csv.suffix == '.gz' else open
    with opener(sifts_csv, 'rt', newline='') as f:
        rows = csv.reader(f)
        header = None
        for row in rows:
            if not row or row[0].startswith('#'):
                continue
            if header is None:
                header = [c.strip() for c in row]
                continue
            rec = dict(zip(header, row))
            pdb = (rec.get('PDB') or '').lower()
            chain = (rec.get('CHAIN') or '').upper()
            acc = (rec.get('SP_PRIMARY') or '').strip()
            if pdb and chain and acc and ((pdb, chain) not in out):
                out[pdb, chain] = acc
    return out

def find_structure(pdb_dir: Path, pdb_id: str) -> Path | None:
    pid = pdb_id.upper()
    for name in (f'{pid}.cif.gz', f'{pid}.cif', f'{pid}.pdb'):
        p = pdb_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    for name in (f'{pdb_id}.cif.gz', f'{pdb_id}.cif', f'{pdb_id}.pdb'):
        p = pdb_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None

def _clean_seq(raw: str) -> str:
    s = raw.replace('\n', '').replace(' ', '').replace(';', '').upper()
    return s

def extract_polymer_chains(path: Path) -> list[tuple[str, str]]:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    if path.suffix == '.gz' or path.name.endswith('.cif.gz'):
        with gzip.open(path, 'rt') as f:
            d = MMCIF2Dict(f)
    elif path.suffix in {'.cif', '.mmcif'} or path.name.endswith('.cif'):
        d = MMCIF2Dict(str(path))
    else:
        return _extract_from_pdb(path)
    types = d.get('_entity_poly.type', [])
    strands = d.get('_entity_poly.pdbx_strand_id', [])
    seqs = d.get('_entity_poly.pdbx_seq_one_letter_code', [])
    if isinstance(types, str):
        types, strands, seqs = ([types], [strands], [seqs])
    if not (types and strands and seqs) or not len(types) == len(strands) == len(seqs):
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for typ, strand, seq in zip(types, strands, seqs):
        if 'polypeptide' not in typ.lower():
            continue
        clean = _clean_seq(seq)
        if not clean:
            continue
        for ch in str(strand).split(','):
            ch = ch.strip().upper()
            if not ch or ch in seen:
                continue
            seen.add(ch)
            out.append((ch, clean))
    return out

def _extract_from_pdb(path: Path) -> list[tuple[str, str]]:
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1_extended
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('x', str(path))
    out: list[tuple[str, str]] = []
    model = next(struct.get_models())
    for chain in model:
        aa: list[str] = []
        for res in chain:
            if not is_aa(res, standard=False):
                continue
            name = res.get_resname().strip().upper()
            aa.append(protein_letters_3to1_extended.get(name, 'X'))
        if aa:
            out.append((chain.id.upper(), ''.join(aa)))
    return out

def pass_filters(seq: str, cfg: dict) -> bool:
    min_len = int(cfg['min_len'])
    max_len = int(cfg['max_len'])
    max_unk = float(cfg['max_unknown_frac'])
    if len(seq) < min_len or len(seq) > max_len:
        return False
    unk = sum((1 for c in seq if c not in AA))
    if len(seq) and unk / len(seq) > max_unk:
        return False
    return True

def stratified_holdout(chain_ids: list[str], folds: dict[str, str], *, fraction: float, max_n: int, seed: int) -> set[str]:
    by_fold: dict[str, list[str]] = defaultdict(list)
    for cid in chain_ids:
        by_fold[folds.get(cid, 'unknown')].append(cid)
    rng = random.Random(seed)
    hold: list[str] = []
    target = min(max_n, max(1, int(round(len(chain_ids) * fraction))))
    folds_sorted = sorted(by_fold.keys())
    rng.shuffle(folds_sorted)
    for fold in folds_sorted:
        ids = list(by_fold[fold])
        rng.shuffle(ids)
        n = int(round(len(ids) * fraction))
        if len(ids) >= 5 and n == 0:
            n = 1
        hold.extend(ids[:n])
    rng.shuffle(hold)
    return set(hold[:target])

def load_pdb_ids(pdb_dir: Path, limit: int | None) -> list[str]:
    ids_file = pdb_dir / 'paper_ids.txt'
    if not ids_file.exists():
        ids_file = pdb_dir / 'pilot_ids.txt'
    if not ids_file.exists():
        raise FileNotFoundError(f'Missing paper_ids.txt in {pdb_dir}')
    ids = [ln.strip().lower() for ln in ids_file.read_text().splitlines() if ln.strip() and (not ln.startswith('#'))]
    if limit is not None:
        ids = ids[:limit]
    return ids

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true', help='Small subset → corpus_smoke/ (safe; does not touch paper lock)')
    ap.add_argument('--pilot', action='store_true', help='Alias for --smoke (compat with download scripts)')
    ap.add_argument('--limit-pdbs', type=int, default=None)
    ap.add_argument('--force-relock', action='store_true', help='Allow rewriting corpus_lock.json (dangerous for paper)')
    args = ap.parse_args()
    smoke = bool(args.smoke or args.pilot)
    cfg = load_yaml('corpus.yaml')
    seed = int(cfg.get('random_seed', 42))
    outdir = resolve_path(cfg['smoke_outdir'] if smoke else cfg['paper_outdir'])
    ensure_dir(outdir)
    lock_path = outdir / 'corpus_lock.json'
    if lock_path.exists() and (not args.force_relock) and (not smoke):
        print(f'ERROR: {lock_path} already exists. Refusing to rebuild paper lock. Use --force-relock only if intentional.', file=sys.stderr)
        return 1
    pdb_dir = raw_dir('pdb')
    scop_dir = raw_dir('scop')
    sifts_dir = raw_dir('sifts')
    limit = args.limit_pdbs
    if smoke and limit is None:
        limit = int(cfg.get('smoke_max_pdbs', 40))
    pdb_ids = load_pdb_ids(pdb_dir, limit)
    print(f"Building corpus mode={('smoke' if smoke else 'paper')} pdbs={len(pdb_ids)} → {outdir}")
    cla = _find_cla(scop_dir)
    print(f'  SCOPe map from {cla.name}')
    scop_map = load_scop_chain_map(cla)
    sifts_path = sifts_dir / 'pdb_chain_uniprot.csv'
    if not sifts_path.exists():
        sifts_path = sifts_dir / 'pdb_chain_uniprot.csv.gz'
    uniprot_map = load_sifts_uniprot(sifts_path) if sifts_path.exists() else {}
    print(f'  SIFTS UniProt pairs: {len(uniprot_map)}')
    rows: list[dict] = []
    skipped = {'no_structure': 0, 'no_polymer': 0, 'filter': 0, 'no_scop': 0, 'parse_error': 0}
    progress = BatchProgress(len(pdb_ids), label='corpus-PDB')
    t0 = time.time()
    for i, pdb_id in enumerate(pdb_ids, 1):
        path = find_structure(pdb_dir, pdb_id)
        if path is None:
            skipped['no_structure'] += 1
            progress.tick(ok=False, force_print=i == len(pdb_ids))
            continue
        try:
            polymers = extract_polymer_chains(path)
        except Exception as e:
            skipped['parse_error'] += 1
            if skipped['parse_error'] <= 10:
                print(f'  parse fail {pdb_id}: {e}', file=sys.stderr)
            progress.tick(ok=False, force_print=i == len(pdb_ids))
            continue
        if not polymers:
            skipped['no_polymer'] += 1
            progress.tick(ok=False, force_print=i == len(pdb_ids))
            continue
        kept_here = 0
        for chain, seq in polymers:
            if not pass_filters(seq, cfg):
                skipped['filter'] += 1
                continue
            scop = scop_map.get((pdb_id, chain))
            if cfg.get('require_scop_fold', True) and (not scop):
                skipped['no_scop'] += 1
                continue
            chain_id = f'{pdb_id}_{chain}'
            rows.append({'chain_id': chain_id, 'pdb_id': pdb_id, 'chain': chain, 'length': len(seq), 'sequence': seq, 'scop_domain': (scop or {}).get('scop_domain', ''), 'scop_sccs': (scop or {}).get('scop_sccs', ''), 'scop_fold': (scop or {}).get('scop_fold', ''), 'scop_class': (scop or {}).get('scop_class', ''), 'uniprot': uniprot_map.get((pdb_id, chain), ''), 'structure_file': str(path.relative_to(BIO_ROOT))})
            kept_here += 1
        progress.tick(ok=kept_here > 0, force_print=i == len(pdb_ids))
    if not rows:
        print('ERROR: no chains kept. Check filters / downloads.', file=sys.stderr)
        return 1
    dedup: dict[str, dict] = {}
    for r in rows:
        dedup.setdefault(r['chain_id'], r)
    rows = list(dedup.values())
    rows.sort(key=lambda r: r['chain_id'])
    chain_ids = [r['chain_id'] for r in rows]
    folds = {r['chain_id']: r['scop_fold'] or 'unknown' for r in rows}
    if smoke:
        max_hold = int(cfg.get('smoke_holdout_max', 8))
    else:
        max_hold = int(cfg.get('paper_holdout_max', 1500))
    fraction = float(cfg.get('holdout_fraction', 0.1))
    holdout = stratified_holdout(chain_ids, folds, fraction=fraction, max_n=max_hold, seed=seed)
    train = [c for c in chain_ids if c not in holdout]
    hold_list = sorted(holdout)
    for r in rows:
        r['split'] = 'holdout' if r['chain_id'] in holdout else 'train'
    fasta = outdir / 'sequences.fasta'
    with fasta.open('w') as f:
        for r in rows:
            f.write(f">{r['chain_id']}\n")
            seq = r['sequence']
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + '\n')
    csv_path = outdir / 'chains.csv'
    fieldnames = ['chain_id', 'pdb_id', 'chain', 'length', 'split', 'scop_domain', 'scop_sccs', 'scop_fold', 'scop_class', 'uniprot', 'structure_file', 'sequence']
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    (outdir / 'split_train.txt').write_text('\n'.join(train) + '\n')
    (outdir / 'split_holdout.txt').write_text('\n'.join(hold_list) + '\n')
    lock = {'mode': 'smoke' if smoke else 'paper', 'random_seed': seed, 'holdout_fraction': fraction, 'holdout_max': max_hold, 'n_chains': len(rows), 'n_train': len(train), 'n_holdout': len(hold_list), 'holdout_chain_ids': hold_list, 'require_scop_fold': bool(cfg.get('require_scop_fold', True)), 'min_len': int(cfg['min_len']), 'max_len': int(cfg['max_len']), 'pdb_ids_used': len(pdb_ids), 'note': 'Held-out IDs are locked. Do not use holdout for tuning thresholds. Smoke locks are disposable; paper lock must not be rewritten.'}
    write_json(lock_path, lock)
    manifest = {'outdir': str(outdir.relative_to(BIO_ROOT)), 'mode': lock['mode'], 'n_chains': len(rows), 'n_train': len(train), 'n_holdout': len(hold_list), 'skipped': skipped, 'elapsed_s': round(time.time() - t0, 1), 'paths': {'chains_csv': str(csv_path.relative_to(BIO_ROOT)), 'sequences_fasta': str(fasta.relative_to(BIO_ROOT)), 'lock': str(lock_path.relative_to(BIO_ROOT))}}
    write_json(outdir / 'build_manifest.json', manifest)
    print(f'Corpus ready: {len(rows)} chains (train={len(train)}, holdout={len(hold_list)}) → {outdir}')
    print(f'  skipped: {skipped}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
