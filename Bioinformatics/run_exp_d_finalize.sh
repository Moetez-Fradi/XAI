#!/usr/bin/env bash
# Finalize Experiment D: integrate corpus → index → pairs → case study.
#
# Prerequisites:
#   ./run_exp_d_downloads.sh   (or at least build_exp_d_supplement.py done)
#   ./start_xqdrant.sh --daemon
#
# Usage:
#   ./run_exp_d_finalize.sh
#   ./run_exp_d_finalize.sh --skip-integrate   # if index already synced
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

export PATH="$ROOT/tools/mamba/envs/bio-tools/bin:$PATH"

# Verify DSSP before long pipeline steps
if ! command -v mkdssp >/dev/null 2>&1 && ! command -v dssp >/dev/null 2>&1 \
   && [ ! -x "$ROOT/tools/mamba/envs/bio-tools/bin/mkdssp" ]; then
  echo "ERROR: mkdssp/dssp not found." >&2
  echo "  Project conda: export MAMBA_ROOT_PREFIX=$ROOT/tools/mamba && \\" >&2
  echo "    $ROOT/tools/bin/micromamba install -y -n bio-tools -c conda-forge -c bioconda dssp" >&2
  echo "  Arch AUR: yay -S dssp   (not: pacman -S dssp)" >&2
  exit 1
fi

if [ ! -x "$PY" ]; then
  echo "ERROR: missing .venv. Run ./setup_env.sh first." >&2
  exit 1
fi

SKIP_INTEGRATE=0
for arg in "$@"; do
  case "$arg" in
    --skip-integrate) SKIP_INTEGRATE=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Usage: $0 [--skip-integrate]" >&2
      exit 1
      ;;
  esac
done

echo "==> Ensure supplement corpus exists"
if [ ! -f data/processed/corpus_supplement/chains.csv ]; then
  if [ ! -f data/raw/thermo/exp_d_pdb_ids.txt ]; then
    echo "ERROR: run ./run_exp_d_downloads.sh first (no supplement PDB list)" >&2
    exit 1
  fi
  "$PY" scripts/build_exp_d_supplement.py
fi

if [ "$SKIP_INTEGRATE" -eq 0 ]; then
  echo ""
  echo "==> Merge corpus + sync embeddings + XQdrant index"
  ./start_xqdrant.sh --daemon 2>/dev/null || true
  "$PY" scripts/integrate_exp_d_supplement.py
fi

echo ""
echo "==> Curate literature pairs (merged corpus)"
"$PY" scripts/curate_exp_d_literature.py

echo ""
echo "==> Build thermo/meso pairs"
"$PY" scripts/prep_exp_d_pairs.py

echo ""
echo "==> Experiment D case study"
"$PY" scripts/exp_d_thermo.py --restart

echo ""
echo "==> Final status"
"$PY" - <<'PY'
import json
from pathlib import Path

root = Path(".")
pairs_m = json.loads((root / "data/processed/exp_d/pairs_manifest.json").read_text())
summary = (root / "results/exp_d/summary.tsv").read_text().strip().splitlines()
sig = (root / "results/exp_d/significance.tsv").read_text().strip().splitlines()
print("pairs:", pairs_m)
print("--- summary.tsv ---")
for line in summary[:25]:
    print(line)
print("--- significance.tsv ---")
for line in sig:
    print(line)
PY

echo ""
echo "Done → results/exp_d/ (Experiment D finalized; plots → plot_scripts/ later)"
