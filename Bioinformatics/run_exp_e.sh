#!/usr/bin/env bash
# Experiment E — functional annotation transfer / active-site localization.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

export PATH="$ROOT/tools/mamba/envs/bio-tools/bin:$PATH"

if [ ! -x "$PY" ]; then
  echo "ERROR: missing .venv. Run ./setup_env.sh first." >&2
  exit 1
fi

PAPER=1
RESTART=""
SKIP_TM=""
FORCE_BENCH=0

while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) PAPER=0 ;;
    --restart) RESTART="--restart" ;;
    --skip-tmalign) SKIP_TM="--skip-tmalign" ;;
    --force-benchmark) FORCE_BENCH=1 ;;
    *)
      echo "Usage: $0 [--smoke] [--restart] [--skip-tmalign] [--force-benchmark]" >&2
      exit 1
      ;;
  esac
  shift
done

SMOKE_FLAG=""
EXP_A="results/exp_a/checkpoints/queries.jsonl"
BENCH="data/processed/exp_e/benchmark_queries.tsv"
if [ "$PAPER" -eq 0 ]; then
  SMOKE_FLAG="--smoke"
  EXP_A="results/exp_a_smoke/checkpoints/queries.jsonl"
  BENCH="data/processed/exp_e/benchmark_queries.tsv"
fi

if [ ! -f "$EXP_A" ]; then
  echo "ERROR: need Exp A results ($EXP_A)." >&2
  exit 1
fi

if [ ! -f data/raw/sifts/pdb_chain_uniprot.csv ] && [ ! -f data/raw/sifts/pdb_chain_uniprot.csv.gz ]; then
  echo "ERROR: run ./run_downloads.sh (SIFTS + UniProt JSON) first." >&2
  exit 1
fi

# UniProt JSON for mapped accessions
if [ ! -d data/raw/uniprot/json ] || [ -z "$(ls -A data/raw/uniprot/json 2>/dev/null | head -1)" ]; then
  echo "==> Fetching UniProt JSON for PDB-mapped accessions..."
  "$PY" scripts/download_uniprot.py
fi

# DSSP annotations (reuse Exp B path)
if [ ! -f data/annotations/dssp_chain_features.csv ]; then
  echo "==> DSSP annotations missing — run annotate_dssp.py (see run_exp_b_v3.sh --backfill-profiles)"
  echo "    Minimum: annotate holdout benchmark chains after prep step."
fi

echo "==> Exp E benchmark curation"
PREP_ARGS="$SMOKE_FLAG"
if [ "$FORCE_BENCH" -eq 1 ]; then
  PREP_ARGS="$PREP_ARGS --force"
fi
"$PY" scripts/prep_exp_e_benchmark.py $PREP_ARGS

if [ ! -f "$BENCH" ]; then
  echo "ERROR: benchmark empty — check UniProt JSON + SIFTS + DSSP" >&2
  exit 1
fi

# Baseline structures for TM-align (optional but recommended)
if [ -z "$SKIP_TM" ] && [ ! -d data/baselines/exp_a/structures ]; then
  echo "==> Exp A baseline structures missing — run ./run_exp_a_baselines.sh --fast-only (or full)"
  echo "    Continuing; TM-align comparison will be skipped if structures absent."
fi

echo "==> Exp E functional localization (paper=$PAPER)"
"$PY" scripts/exp_e_functional.py $SMOKE_FLAG $RESTART $SKIP_TM

echo ""
if [ "$PAPER" -eq 0 ]; then
  echo "Artifacts: results/exp_e_smoke/"
else
  echo "Artifacts: results/exp_e/"
fi
echo "  metrics.json  summary.tsv  significance.tsv  localization_comparison.tsv"
echo "Plots: cd plot_scripts && python plot_exp_e_functional.py"
echo "PyMOL: python generate_pymol_exp_e.py"
