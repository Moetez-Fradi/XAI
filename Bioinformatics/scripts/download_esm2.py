"""Download ESM2 checkpoint (PyTorch/safetensors only) into models/."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import BIO_ROOT, download_cfg, ensure_dir, load_yaml, write_json

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-id', default=None)
    ap.add_argument('--local-dir', type=Path, default=None)
    args = ap.parse_args()
    dl = download_cfg().get('esm2', {})
    esm = load_yaml('esm2.yaml')
    repo_id = args.repo_id or dl.get('repo_id') or esm.get('model_id')
    local_rel = args.local_dir or Path(dl.get('local_dir') or esm.get('local_dir'))
    local_dir = local_rel if local_rel.is_absolute() else BIO_ROOT / local_rel
    ensure_dir(local_dir)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print('huggingface_hub not installed. Run ./setup_env.sh first.', file=sys.stderr)
        return 1
    allow = dl.get('allow_patterns')
    ignore = dl.get('ignore_patterns')
    print(f'Downloading {repo_id} → {local_dir}')
    print('  (PyTorch/safetensors only — skipping tf_model.h5)')
    print('  Hugging Face shows its own progress bar / transfer stats below.')
    path = snapshot_download(repo_id=repo_id, local_dir=str(local_dir), allow_patterns=allow, ignore_patterns=ignore)
    write_json(local_dir / 'download_manifest.json', {'repo_id': repo_id, 'local_dir': path, 'allow_patterns': allow, 'ignore_patterns': ignore})
    print(f'ESM2 ready at {path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
