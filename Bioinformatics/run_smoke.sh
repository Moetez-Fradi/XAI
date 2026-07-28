#!/usr/bin/env bash
# End-to-end smoke: build a tiny corpus + embed with local ESM2.
# Does NOT touch the paper corpus lock (writes under corpus_smoke/ + embeddings/smoke/).
#
# Usage (from Bioinformatics/):
#   source .venv/bin/activate   # optional; script uses .venv/bin/python
#   ./run_smoke.sh
#   ./run_smoke.sh --limit-pdbs 20 --embed-limit 16
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LIMIT_PDBS=40
EMBED_LIMIT=32
DEVICE_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --limit-pdbs) LIMIT_PDBS="$2"; shift 2 ;;
    --embed-limit) EMBED_LIMIT="$2"; shift 2 ;;
    --device) DEVICE_ARGS=(--device "$2"); shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: missing .venv. Run ./setup_env.sh first." >&2
  exit 1
fi

if [ ! -f models/esm2_t33_650M_UR50D/config.json ]; then
  echo "ERROR: ESM2 not found under models/. Run ./run_downloads.sh first." >&2
  exit 1
fi

if [ ! -f data/raw/pdb/paper_ids.txt ] && [ ! -f data/raw/pdb/pilot_ids.txt ]; then
  echo "ERROR: no PDB id list. Run ./run_downloads.sh first." >&2
  exit 1
fi

echo "==> Smoke corpus (≤${LIMIT_PDBS} PDBs) → data/processed/corpus_smoke/"
"$PY" scripts/build_corpus.py --smoke --limit-pdbs "$LIMIT_PDBS"

echo ""
echo "==> Smoke embed (≤${EMBED_LIMIT} chains) → embeddings/smoke/"
"$PY" scripts/embed_esm2.py --smoke --limit "$EMBED_LIMIT" \
  ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"}

echo ""
"$PY" - <<'PY'
from pathlib import Path
import json
import numpy as np

root = Path(".")
vec = np.load(root / "embeddings/smoke/vectors.npy")
ids = (root / "embeddings/smoke/ids.txt").read_text().splitlines()
meta = json.loads((root / "embeddings/smoke/meta.json").read_text())
lock = json.loads((root / "data/processed/corpus_smoke/corpus_lock.json").read_text())
assert vec.ndim == 2 and vec.shape[0] == len(ids) == meta["n"]
assert np.isfinite(vec).all()
assert lock["n_holdout"] >= 1
print("SMOKE OK")
print(f"  corpus chains: {lock['n_chains']} (train={lock['n_train']} holdout={lock['n_holdout']})")
print(f"  embeddings:    {vec.shape}  dim={meta['dim']}  device={meta['device']}")
print(f"  sample ids:    {', '.join(ids[:5])}")
print("")
print("Next (full paper corpus — long):")
print("  .venv/bin/python scripts/build_corpus.py")
print("  .venv/bin/python scripts/embed_esm2.py          # after corpus; needs GPU recommended")
PY
