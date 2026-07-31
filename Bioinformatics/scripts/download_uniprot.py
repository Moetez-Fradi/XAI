"""Download SIFTS maps, Swiss-Prot, and UniProt JSON for paper PDB accessions.

Writes:
  data/raw/sifts/*.csv[.gz]
  data/raw/uniprot/uniprot_sprot.dat.gz
  data/raw/uniprot/uniprot_sprot.fasta.gz
  data/raw/uniprot/json/<ACCESSION>.json   (batch REST for PDB-mapped IDs)
  data/raw/uniprot/accessions.txt
  data/raw/uniprot/download_manifest.json
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import BatchProgress, download_cfg, download_file, ensure_dir, gunzip_to, raw_dir, write_json

def _load_pdb_ids(pdb_dir: Path) -> set[str]:
    for name in ('paper_ids.txt', 'pilot_ids.txt'):
        ids_file = pdb_dir / name
        if ids_file.exists():
            return {ln.strip().lower() for ln in ids_file.read_text().splitlines() if ln.strip() and (not ln.startswith('#'))}
    raise FileNotFoundError(f'Missing paper_ids.txt / pilot_ids.txt in {pdb_dir}. Run download_pdb.py first.')

def _accessions_for_pdbs(sifts_csv: Path, pdb_ids: set[str]) -> list[str]:
    accs: dict[str, None] = {}
    with sifts_csv.open(newline='') as f:
        rows = csv.reader(f)
        header = None
        for row in rows:
            if not row or row[0].startswith('#'):
                continue
            if header is None:
                header = [c.strip() for c in row]
                continue
            rec = dict(zip(header, row))
            pdb = (rec.get('PDB') or rec.get('pdb') or '').lower()
            acc = (rec.get('SP_PRIMARY') or rec.get('ACCESSION') or '').strip()
            if pdb in pdb_ids and acc:
                accs[acc] = None
    return list(accs.keys())

def _batch_fetch(accessions: list[str], dest_dir: Path, force: bool, *, batch_size: int=100, sleep_s: float=0.15) -> tuple[int, list]:
    ensure_dir(dest_dir)
    ok, fail = (0, [])
    pending = [a for a in accessions if force or not (dest_dir / f'{a}.json').exists() or (dest_dir / f'{a}.json').stat().st_size == 0]
    cached = len(accessions) - len(pending)
    print(f'  batch REST: {len(pending)} to fetch ({cached} cached)')
    progress = BatchProgress(len(accessions), label='UniProt-JSON')
    for _ in range(cached):
        progress.tick(skipped=True, every_s=999)
    for i in range(0, len(pending), batch_size):
        chunk = pending[i:i + batch_size]
        qs = ','.join(chunk)
        url = f'https://rest.uniprot.org/uniprotkb/accessions?accessions={qs}&format=json&size={len(chunk)}'
        req = Request(url, headers={'Accept': 'application/json', 'User-Agent': 'XQdrant-Bioinformatics/0.1'})
        try:
            with urlopen(req, timeout=180) as resp:
                raw = resp.read()
                payload = json.loads(raw.decode())
            results = payload.get('results') or payload
            if isinstance(results, dict):
                results = [results]
            got: set[str] = set()
            for entry in results:
                acc = entry.get('primaryAccession') or ''
                if not acc:
                    continue
                blob = json.dumps(entry) + '\n'
                (dest_dir / f'{acc}.json').write_text(blob)
                got.add(acc)
                ok += 1
                progress.tick(ok=True, nbytes=len(blob.encode()))
            for a in chunk:
                if a not in got and (not (dest_dir / f'{a}.json').exists()):
                    fail.append({'accession': a, 'error': 'missing in batch response'})
                    progress.tick(ok=False)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            print(f'  batch fail ({e}); falling back to singles for {len(chunk)} ids')
            for a in chunk:
                try:
                    dest = dest_dir / f'{a}.json'
                    _single(a, dest, force=True)
                    ok += 1
                    progress.tick(ok=True, nbytes=dest.stat().st_size if dest.exists() else 0)
                except Exception as e2:
                    fail.append({'accession': a, 'error': str(e2)})
                    progress.tick(ok=False)
        if sleep_s:
            time.sleep(sleep_s)
    progress.print_status()
    return (ok, fail)

def _single(accession: str, dest: Path, force: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0 and (not force):
        return
    url = f'https://rest.uniprot.org/uniprotkb/{accession}.json'
    req = Request(url, headers={'User-Agent': 'XQdrant-Bioinformatics/0.1'})
    with urlopen(req, timeout=60) as resp:
        dest.write_text(resp.read().decode())

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--max-accessions', type=int, default=None)
    ap.add_argument('--sifts-only', action='store_true', help='Only SIFTS + Swiss-Prot bulk files; skip per-accession JSON')
    ap.add_argument('--skip-sprot', action='store_true', help='Skip large Swiss-Prot .dat/.fasta downloads')
    args = ap.parse_args()
    cfg = download_cfg()['uniprot']
    sifts_dir = raw_dir('sifts')
    uni_dir = raw_dir('uniprot')
    json_dir = ensure_dir(uni_dir / 'json')
    mode = 'pilot' if args.pilot else 'paper'
    sifts_files = []
    for item in cfg.get('sifts') or [{'name': 'pdb_chain_uniprot.csv.gz', 'url': cfg.get('sifts_url')}]:
        if not item.get('url'):
            continue
        dest = sifts_dir / item['name']
        download_file(item['url'], dest, force=args.force, timeout=600)
        if dest.name.endswith('.gz'):
            gunzip_to(dest)
        sifts_files.append(item['name'])
    if not args.skip_sprot:
        if cfg.get('sprot_dat_gz'):
            download_file(cfg['sprot_dat_gz'], uni_dir / 'uniprot_sprot.dat.gz', force=args.force, timeout=600)
        if cfg.get('sprot_fasta_gz'):
            download_file(cfg['sprot_fasta_gz'], uni_dir / 'uniprot_sprot.fasta.gz', force=args.force, timeout=600)
    if args.sifts_only:
        write_json(uni_dir / 'download_manifest.json', {'mode': mode, 'sifts_only': True, 'sifts': sifts_files})
        print(f'SIFTS (+ optional Swiss-Prot) ready → {sifts_dir}, {uni_dir}')
        return 0
    csv_path = sifts_dir / 'pdb_chain_uniprot.csv'
    if not csv_path.exists():
        gz = sifts_dir / 'pdb_chain_uniprot.csv.gz'
        csv_path = gunzip_to(gz, csv_path)
    pdb_ids = _load_pdb_ids(raw_dir('pdb'))
    accessions = _accessions_for_pdbs(csv_path, pdb_ids)
    if args.max_accessions is not None:
        limit = args.max_accessions
    elif args.pilot:
        limit = int(cfg.get('pilot_max_accessions', 500))
    else:
        limit = int(cfg.get('paper_max_accessions', 0))
    if limit > 0:
        accessions = accessions[:limit]
    (uni_dir / 'accessions.txt').write_text('\n'.join(accessions) + '\n')
    print(f'UniProt JSON for {len(accessions)} accessions (mode={mode})')
    ok, fail = _batch_fetch(accessions, json_dir, args.force, batch_size=int(cfg.get('batch_size', 100)), sleep_s=float(cfg.get('sleep_s', 0.15)))
    write_json(uni_dir / 'download_manifest.json', {'source': 'UniProt REST + SIFTS + Swiss-Prot FTP', 'mode': mode, 'sifts': sifts_files, 'requested': len(accessions), 'ok': ok, 'failed_count': len(fail), 'failed': fail[:50], 'json_dir': str(json_dir), 'outdir': str(uni_dir)})
    print(f'UniProt done: {ok} ok, {len(fail)} failed → {uni_dir}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
