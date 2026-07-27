#!/usr/bin/env bash
# Run paper-scale data (+ ESM2) downloads in order (PSB-level defaults).
#
# Usage (from Bioinformatics/):
#   ./setup_env.sh
#   ./run_downloads.sh              # paper corpus (~10k PDBs + Swiss-Prot + ESM2)
#   ./run_downloads.sh --pilot      # small smoke (~200 PDBs)
#   ./run_downloads.sh --skip-esm2
#   ./run_downloads.sh --force
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SKIP_ESM2=0
SKIP_PDB=0
SKIP_UNIPROT=0
SKIP_THERMO=0
PILOT=0
FORCE_ARGS=()
MODE_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --pilot) PILOT=1; MODE_ARGS=(--pilot) ;;
    --skip-esm2) SKIP_ESM2=1 ;;
    --skip-pdb) SKIP_PDB=1 ;;
    --skip-uniprot) SKIP_UNIPROT=1 ;;
    --skip-thermo) SKIP_THERMO=1 ;;
    --force) FORCE_ARGS=(--force) ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: missing .venv. Run ./setup_env.sh first." >&2
  exit 1
fi

if [ "$PILOT" -eq 1 ]; then
  echo "Mode: PILOT (smoke) — expect ~3–5 GB"
else
  echo "Mode: PAPER (PSB-level)"
  echo "  Network download ≈ 8–15 GB (PDB+SwissProt+ESM2+annotations)"
  echo "  ~25–35 GB free disk recommended later (Foldseek/results), not download size"
  echo "  Wall time: typically several hours; each stage prints % and ETA"
fi

run() {
  local label="$1"
  shift
  echo ""
  echo "======== $label ========"
  local t0=$SECONDS
  "$PY" "$@"
  local dt=$((SECONDS - t0))
  echo "-------- finished $label in ${dt}s --------"
}

run "1/7 SCOPe + ASTRAL" scripts/download_scop.py ${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"}
run "2/7 CATH"           scripts/download_cath.py ${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"}

if [ "$SKIP_PDB" -eq 0 ]; then
  run "3/7 PDB structures" scripts/download_pdb.py \
    ${MODE_ARGS[@]+"${MODE_ARGS[@]}"} ${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"}
else
  echo "======== 3/7 PDB (skipped) ========"
fi

if [ "$SKIP_UNIPROT" -eq 0 ]; then
  run "4/7 UniProt + SIFTS + Swiss-Prot" scripts/download_uniprot.py \
    ${MODE_ARGS[@]+"${MODE_ARGS[@]}"} ${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"}
else
  echo "======== 4/7 UniProt (skipped) ========"
fi

run "5/7 Pfam" scripts/download_pfam.py ${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"}

if [ "$SKIP_THERMO" -eq 0 ]; then
  run "6/7 OMA thermo pairs" scripts/download_oma_thermo.py \
    ${MODE_ARGS[@]+"${MODE_ARGS[@]}"}
else
  echo "======== 6/7 OMA thermo (skipped) ========"
fi

if [ "$SKIP_ESM2" -eq 0 ]; then
  run "7/7 ESM2 model" scripts/download_esm2.py
else
  echo "======== 7/7 ESM2 (skipped) ========"
fi

echo ""
echo "All requested downloads finished."
echo "  raw data:  $ROOT/data/raw/"
echo "  ESM2:      $ROOT/models/esm2_t33_650M_UR50D/"
echo "Next: ./fetch_tools_macos.sh  OR  ./fetch_tools_linux.sh"
