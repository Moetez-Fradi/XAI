#!/usr/bin/env bash
# Experiment C — negative controls / ROC (does not overwrite A/B results).
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

while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) PAPER=0 ;;
    --restart) RESTART="--restart" ;;
    *)
      echo "Usage: $0 [--smoke] [--restart]" >&2
      exit 1
      ;;
  esac
  shift
done

if [ "$PAPER" -eq 0 ]; then
  if [ ! -f results/exp_a_smoke/checkpoints/queries.jsonl ]; then
    echo "ERROR: need Exp A smoke. Run ./run_exp_a_smoke.sh first." >&2
    exit 1
  fi
  echo "==> Exp C (smoke)"
  "$PY" scripts/exp_c_negative_controls.py --smoke $RESTART --max-queries 20
  echo "Done: results/exp_c_smoke/"
  exit 0
fi

if [ ! -f results/exp_a/checkpoints/queries.jsonl ]; then
  echo "ERROR: need Exp A paper results." >&2
  exit 1
fi

echo "==> Exp C negative controls (paper)"
"$PY" scripts/exp_c_negative_controls.py $RESTART

echo ""
echo "Artifacts: results/exp_c/"
echo "  metrics.json  summary.tsv  roc_curves.tsv"
echo "Paper section: paper/experiment_c.md"
