#!/usr/bin/env bash
# Experiment D — download thermo/meso structures, build supplement, integrate.
#
# Brings OMA-mapped enzyme orthologs into the searchable corpus without
# rewriting the paper holdout lock.
#
# Usage (from Bioinformatics/):
#   ./run_exp_d_downloads.sh              # full pipeline
#   ./run_exp_d_downloads.sh --resolve-only   # UniProt→PDB map only
#   ./run_exp_d_downloads.sh --skip-integrate # download + build supplement only
#   ./run_exp_d_downloads.sh --force
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

export PATH="$ROOT/tools/mamba/envs/bio-tools/bin:$PATH"

if [ ! -x "$PY" ]; then
  echo "ERROR: missing .venv. Run ./setup_env.sh first." >&2
  exit 1
fi

FORCE=0
RESOLVE_ONLY=0
SKIP_INTEGRATE=0
SKIP_OMA=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --resolve-only) RESOLVE_ONLY=1 ;;
    --skip-integrate) SKIP_INTEGRATE=1 ;;
    --skip-oma) SKIP_OMA=1 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "Usage: $0 [--force] [--resolve-only] [--skip-integrate] [--skip-oma]" >&2
      exit 1
      ;;
  esac
done

FORCE_ARGS=()
[ "$FORCE" -eq 1 ] && FORCE_ARGS=(--force)

echo "==> 1/5 Ensure OMA thermophile ortholog list (all seeds)"
if [ "$SKIP_OMA" -eq 0 ]; then
  "$PY" scripts/download_oma_thermo.py
else
  echo "    (skipped)"
fi

echo ""
echo "==> 2/5 Resolve UniProt → PDB + download structures"
RESOLVE_ARGS=()
[ "$RESOLVE_ONLY" -eq 1 ] && RESOLVE_ARGS=(--skip-download)
"$PY" scripts/download_exp_d_structures.py "${FORCE_ARGS[@]}" "${RESOLVE_ARGS[@]}"

if [ "$RESOLVE_ONLY" -eq 1 ]; then
  echo "Resolve-only complete. See data/raw/thermo/exp_d_structure_manifest.json"
  exit 0
fi

echo ""
echo "==> 3/5 Build supplement corpus from Exp D PDBs"
"$PY" scripts/build_exp_d_supplement.py

if [ "$SKIP_INTEGRATE" -eq 1 ]; then
  echo ""
  echo "Skip integrate. Next: ./integrate_exp_d_supplement.py (needs GPU/time for embed)"
  exit 0
fi

echo ""
echo "==> 4/5 Merge corpus + append embeddings + resume XQdrant index"
echo "    (requires ./start_xqdrant.sh --daemon; embedding may take a while on CPU)"
./start_xqdrant.sh --daemon 2>/dev/null || true
"$PY" scripts/integrate_exp_d_supplement.py

echo ""
echo "==> 5/5 Rebuild Exp D pairs + rerun case study"
"$PY" scripts/curate_exp_d_literature.py
"$PY" scripts/prep_exp_d_pairs.py
"$PY" scripts/exp_d_thermo.py --restart

echo ""
echo "Done."
echo "  manifest: data/raw/thermo/exp_d_structure_manifest.json"
echo "  supplement: data/processed/corpus_supplement/"
echo "  merged:     data/processed/corpus_merged/"
echo "  results:    results/exp_d/"
