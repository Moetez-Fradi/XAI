"""Download Pfam clan + PDB map files → data/raw/pfam/."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import download_cfg, download_file, gunzip_to, raw_dir, write_json

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    cfg = download_cfg()['pfam']
    out = raw_dir('pfam')
    files_meta = []
    for item in cfg['files']:
        dest = out / item['name']
        timeout = 600 if 'regions' in item['name'] else 300
        download_file(item['url'], dest, force=args.force, timeout=timeout, max_retries=5)
        entry = {'name': item['name'], 'url': item['url']}
        if dest.name.endswith('.gz'):
            try:
                plain = gunzip_to(dest)
                entry['uncompressed'] = plain.name
            except OSError as e:
                entry['decompress_error'] = str(e)
        files_meta.append(entry)
    write_json(out / 'download_manifest.json', {'source': 'Pfam FTP', 'files': files_meta, 'outdir': str(out)})
    print(f'Pfam ready under {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
