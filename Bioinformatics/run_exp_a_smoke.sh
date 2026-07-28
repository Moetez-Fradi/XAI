#!/usr/bin/env bash
# Smoke: ensure XQdrant is up → index smoke embeddings → Experiment A.
#
# Prerequisites: ./run_smoke.sh already produced embeddings/smoke/
#
# Usage:
#   ./start_xqdrant.sh --daemon
#   ./run_exp_a_smoke.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "ERROR: missing .venv" >&2
  exit 1
fi

if [ ! -f embeddings/smoke/vectors.npy ]; then
  echo "ERROR: missing embeddings/smoke — run ./run_smoke.sh first" >&2
  exit 1
fi

# Re-embed full smoke corpus if holdout chains are missing (old limit=32 runs)
HOLD_N="$("$PY" - <<'PY'
import csv
from pathlib import Path
meta = list(csv.DictReader((Path("data/processed/corpus_smoke/chains.csv")).open()))
hold = {r["chain_id"] for r in meta if r["split"]=="holdout"}
ids = set(Path("embeddings/smoke/ids.txt").read_text().split())
print(len(hold & ids))
PY
)"
if [ "$HOLD_N" -eq 0 ]; then
  echo "==> Re-embedding full smoke corpus (holdout missing from embeddings/smoke)"
  "$PY" scripts/embed_esm2.py --smoke --device cuda 2>/dev/null \
    || "$PY" scripts/embed_esm2.py --smoke
fi

echo "==> Index smoke → XQdrant"
"$PY" scripts/index_xqdrant.py --smoke --restart

echo "==> Experiment A (smoke)"
"$PY" scripts/exp_a_retrieval.py --smoke --restart

echo ""
echo "Smoke Exp A done. Paper path (resumes checkpoints; add --restart to wipe):"
echo "  ./start_xqdrant.sh --daemon   # if not already"
echo "  .venv/bin/python scripts/index_xqdrant.py"
echo "  .venv/bin/python scripts/exp_a_retrieval.py"
