"""Build a starter thermophile/mesophile ortholog pair list via OMA REST API.

Uses seed UniProt accessions from configs/download.yaml, queries OMA for 1:1
orthologs, and writes a CSV under data/raw/thermo/. This is a bootstrap for
Experiment D — expand / replace with literature-curated pairs later.

Writes:
  data/raw/thermo/oma_ortholog_pairs.csv
  data/raw/thermo/download_manifest.json
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
from common_bio import download_cfg, raw_dir, write_json
OMA = 'https://omabrowser.org/api'

def _oma_get(path: str) -> dict | list:
    url = f'{OMA}{path}'
    req = Request(url, headers={'Accept': 'application/json', 'User-Agent': 'XQdrant-Bio/0.1'})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())

def _protein_by_uniprot(acc: str) -> dict | None:
    try:
        data = _oma_get(f'/protein/{acc}/')
        if isinstance(data, dict):
            return data
    except HTTPError as e:
        if e.code == 404:
            return None
        raise
    return None

def _orthologs(entry_id: str, rel_type: str='1:1') -> list[dict]:
    try:
        data = _oma_get(f'/protein/{entry_id}/orthologs/?rel_type={rel_type}')
        return data if isinstance(data, list) else []
    except HTTPError:
        return []

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-pairs', type=int, default=None, help='Global cap (legacy)')
    ap.add_argument('--max-pairs-per-seed', type=int, default=None, help='Max thermophile ortholog rows per seed (default: from download.yaml)')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--sleep', type=float, default=0.25)
    args = ap.parse_args()
    cfg = download_cfg()['thermo']
    out = raw_dir('thermo')
    seeds = list(cfg.get('seed_uniprot') or [])
    if args.max_pairs is not None:
        max_pairs = args.max_pairs
        max_per_seed = max_pairs
    elif args.max_pairs_per_seed is not None:
        max_per_seed = args.max_pairs_per_seed
        max_pairs = max_per_seed * max(len(seeds), 1)
    elif args.pilot:
        max_per_seed = int(cfg.get('pilot_max_pairs_per_seed', 10))
        max_pairs = int(cfg.get('pilot_max_pairs', cfg.get('max_pairs', 50)))
    else:
        max_per_seed = int(cfg.get('paper_max_pairs_per_seed', 15))
        max_pairs = int(cfg.get('paper_max_pairs', cfg.get('max_pairs', 200)))
    meso_code = str(cfg.get('meso_species', 'ECOLI')).upper()
    thermo_hint = str(cfg.get('thermo_species', 'THERM')).upper()
    rows: list[dict] = []
    errors: list[dict] = []

    def _is_thermo_ortholog(o: dict, seed_species: str) -> bool:
        o_species = (o.get('species') or {}).get('code') or ''
        o_name = (o.get('species') or {}).get('species') or ''
        code = o_species.upper()
        name = o_name.upper()
        if seed_species.upper() != meso_code:
            return False
        if code.startswith('THE') or code.startswith('THER'):
            return True
        return 'THERM' in code or 'THERM' in name or 'THERMUS' in name
    for seed in seeds:
        if len(rows) >= max_pairs:
            break
        seed_rows_before = len(rows)
        print(f'OMA lookup seed {seed}')
        try:
            prot = _protein_by_uniprot(seed)
            if not prot:
                errors.append({'seed': seed, 'error': 'not found in OMA'})
                continue
            oma_id = prot.get('omaid') or prot.get('canonicalid') or seed
            species = (prot.get('species') or {}).get('code') or ''
            orths = _orthologs(str(oma_id), rel_type='1:1')
            for o in orths:
                if len(rows) - seed_rows_before >= max_per_seed:
                    break
                if len(rows) >= max_pairs:
                    break
                if not _is_thermo_ortholog(o, species):
                    continue
                o_species = (o.get('species') or {}).get('code') or ''
                o_name = (o.get('species') or {}).get('species') or ''
                o_xrefs = o.get('canonicalid') or o.get('omaid') or ''
                rows.append({'seed_uniprot': seed, 'seed_oma': oma_id, 'seed_species': species, 'ortholog_id': o_xrefs, 'ortholog_oma': o.get('omaid', ''), 'ortholog_species': o_species, 'ortholog_species_name': o_name, 'rel_type': '1:1'})
            time.sleep(args.sleep)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            errors.append({'seed': seed, 'error': str(e)})
            print(f'  FAIL {seed}: {e}', file=sys.stderr)
    csv_path = out / 'oma_ortholog_pairs.csv'
    fieldnames = ['seed_uniprot', 'seed_oma', 'seed_species', 'ortholog_id', 'ortholog_oma', 'ortholog_species', 'ortholog_species_name', 'rel_type']
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    stub = out / 'literature_pairs.stub.csv'
    if not stub.exists():
        stub.write_text('uniprot_thermo,uniprot_meso,pdb_thermo,pdb_meso,note,citation\n# Add curated pairs here for Experiment D\n')
    write_json(out / 'download_manifest.json', {'source': 'OMA REST API', 'pairs': len(rows), 'errors': errors, 'csv': str(csv_path), 'meso_species': meso_code, 'thermo_hint': thermo_hint})
    print(f'Thermo pairs: {len(rows)} written → {csv_path}')
    if not rows:
        print('Note: 0 pairs matched filters; check OMA connectivity or edit seeds in download.yaml.', file=sys.stderr)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
