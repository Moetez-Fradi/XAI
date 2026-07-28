#!/usr/bin/env bash
# Exp A baselines — safe to run while Exp B is DSSP/permuting (CPU).
#
# Usage:
#   source tools/env_linux.sh
#   ./run_exp_a_baselines.sh              # paper
#   ./run_exp_a_baselines.sh --smoke
#   ./run_exp_a_baselines.sh --fast-only   # FASTA prep + ESM2 + BLAST (no structures)
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
export PATH="$ROOT/tools/mamba/envs/bio-tools/bin:$PATH"

SMOKE=0
FAST_ONLY=0
RESTART_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --fast-only) FAST_ONLY=1 ;;
    --restart) RESTART_ARGS=(--restart) ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown: $arg" >&2; exit 1 ;;
  esac
done

MODE_ARGS=()
[ "$SMOKE" -eq 1 ] && MODE_ARGS=(--smoke)

echo "==> Prep baseline inputs"
if [ "$FAST_ONLY" -eq 1 ]; then
  "$PY" scripts/prep_exp_a_baselines.py "${MODE_ARGS[@]}" ${RESTART_ARGS[@]+"${RESTART_ARGS[@]}"} --skip-structures
else
  "$PY" scripts/prep_exp_a_baselines.py "${MODE_ARGS[@]}" ${RESTART_ARGS[@]+"${RESTART_ARGS[@]}"}
fi

echo ""
echo "==> ESM2 cosine + BLASTp in parallel"
"$PY" scripts/exp_a_baseline_esm2.py "${MODE_ARGS[@]}" ${RESTART_ARGS[@]+"${RESTART_ARGS[@]}"} &
PID_ESM=$!
"$PY" scripts/exp_a_baseline_blast.py "${MODE_ARGS[@]}" ${RESTART_ARGS[@]+"${RESTART_ARGS[@]}"} &
PID_BLAST=$!
wait $PID_ESM
wait $PID_BLAST

if [ "$FAST_ONLY" -eq 1 ]; then
  echo "Fast-only done (no Foldseek/TM-align). Aggregate partial:"
  "$PY" scripts/exp_a_aggregate_baselines.py "${MODE_ARGS[@]}"
  exit 0
fi

echo ""
echo "==> Foldseek"
"$PY" scripts/exp_a_baseline_foldseek.py "${MODE_ARGS[@]}" ${RESTART_ARGS[@]+"${RESTART_ARGS[@]}"}

echo ""
echo "==> TM-align re-rank (Foldseek candidates)"
"$PY" scripts/exp_a_baseline_tmalign.py "${MODE_ARGS[@]}" ${RESTART_ARGS[@]+"${RESTART_ARGS[@]}"}

echo ""
echo "==> Aggregate comparison table"
"$PY" scripts/exp_a_aggregate_baselines.py "${MODE_ARGS[@]}"

echo ""
echo "Baselines done → results/exp_a/baselines/comparison.tsv"
