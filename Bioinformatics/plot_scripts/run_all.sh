#!/usr/bin/env bash
# Generate all paper figures from completed experiments A–E.
#
# Usage:
#   cd Bioinformatics
#   ./plot_scripts/run_all.sh
#
# Outputs → plot_scripts/outputs/*.png|.pdf
# PyMOL   → plot_scripts/outputs/pymol/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
PS="$ROOT/plot_scripts"

if [ ! -x "$PY" ]; then
  echo "ERROR: missing .venv — run ./setup_env.sh" >&2
  exit 1
fi

echo "==> Experiment A — retrieval"
"$PY" "$PS/plot_exp_a_retrieval.py"

echo ""
echo "==> Experiment B v3 — attribution validity"
"$PY" "$PS/plot_exp_b_v3_validity.py"

echo ""
echo "==> Experiment C — ROC"
"$PY" "$PS/plot_exp_c_roc.py"

echo ""
echo "==> Experiment D — thermostability"
"$PY" "$PS/plot_exp_d_thermo.py"

echo ""
echo "==> Experiment D — PyMOL sessions"
"$PY" "$PS/generate_pymol_exp_d.py"

echo ""
echo "==> Experiment E — active-site localization"
if [ -f "$ROOT/results/exp_e/metrics.json" ]; then
  "$PY" "$PS/plot_exp_e_functional.py"
  "$PY" "$PS/generate_pymol_exp_e.py"
else
  echo "  (skip — run ./run_exp_e.sh first)"
fi

echo ""
echo "==> Experiment F — workflow timing"
if [ -f "$ROOT/results/exp_f/metrics.json" ]; then
  "$PY" "$PS/plot_exp_f_workflow.py"
else
  echo "  (skip — run ./run_exp_f.sh first)"
fi

echo ""
echo "==> Pipeline schematic + ablations"
"$PY" "$PS/plot_pipeline_schematic.py"
if [ -f "$ROOT/results/exp_ablation/metrics.json" ]; then
  "$PY" "$PS/plot_exp_ablation.py"
else
  echo "  (skip ablation plot — run ./run_exp_ablation.sh first)"
fi

echo ""
echo "Done → plot_scripts/outputs/"
ls -la "$PS/outputs/" 2>/dev/null | tail -20
