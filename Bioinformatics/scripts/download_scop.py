"""Download SCOPe parseable files + ASTRAL sequence subsets → data/raw/scop/."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import download_cfg, download_file, raw_dir, write_json

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--force', action='store_true', help='Re-download even if present')
    args = ap.parse_args()
    cfg = download_cfg()['scop']
    out = raw_dir('scop')
    files: list[str] = []
    for item in cfg['files']:
        dest = out / item['name']
        download_file(item['url'], dest, force=args.force, timeout=300)
        files.append(item['name'])
    for item in cfg.get('astral') or []:
        dest = out / item['name']
        download_file(item['url'], dest, force=args.force, timeout=600)
        files.append(item['name'])
    write_json(out / 'download_manifest.json', {'source': 'SCOPe + ASTRAL', 'version': cfg.get('version'), 'files': files, 'outdir': str(out)})
    print(f'SCOPe/ASTRAL ready under {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
