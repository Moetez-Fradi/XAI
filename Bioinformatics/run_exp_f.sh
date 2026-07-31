#!/usr/bin/env bash
# Experiment F — runtime / workflow comparison (steps.md §4F).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
PS="$ROOT/plot_scripts"

export PATH="$ROOT/tools/blast/bin:$ROOT/tools/mamba/envs/bio-tools/bin:$PATH"

if [ ! -x "$PY" ]; then
  echo "ERROR: missing .venv. Run ./setup_env.sh first." >&2
  exit 1
fi

PAPER=1
RESTART=""
SKIP_EMB=""
MAX_Q=""
INDEXED=0
PLOT=0

while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) PAPER=0 ;;
    --restart) RESTART="--restart" ;;
    --skip-embed) SKIP_EMB="--skip-embed" ;;
    --indexed) INDEXED=1 ;;
    --plot) PLOT=1 ;;
    --complete)
      RESTART="--restart"
      INDEXED=1
      PLOT=1
      ;;
    --max-queries)
      shift
      MAX_Q="--max-queries $1"
      ;;
    *)
      echo "Usage: $0 [--smoke] [--restart] [--skip-embed] [--indexed] [--plot] [--complete] [--max-queries N]" >&2
      exit 1
      ;;
  esac
  shift
done

SMOKE_FLAG=""
BASE="data/baselines/exp_a"
if [ "$PAPER" -eq 0 ]; then
  SMOKE_FLAG="--smoke"
  BASE="data/baselines/exp_a_smoke"
fi

if [ "$INDEXED" -eq 0 ]; then
  if [ ! -d "$BASE/structures" ] || [ ! -f "$BASE/fasta/holdout.fasta" ]; then
    echo "ERROR: need Exp A baseline prep (structures + FASTA)." >&2
    echo "  ./run_exp_a_baselines.sh" >&2
    exit 1
  fi
fi

if ! curl -sf http://127.0.0.1:6333/readyz >/dev/null 2>&1; then
  echo "==> Starting XQdrant (daemon)..."
  ./start_xqdrant.sh --daemon
  sleep 2
fi

if [ "$INDEXED" -eq 1 ] && [ -n "$RESTART" ] && [ -z "$SKIP_EMB" ] && [ "$PLOT" -eq 1 ]; then
  echo "==> Exp F full run (cold path + traditional baselines)"
  "$PY" scripts/exp_f_runtime.py $SMOKE_FLAG --restart $MAX_Q
  echo ""
  echo "==> Exp F indexed pass (search-only / production latency)"
  "$PY" scripts/exp_f_runtime.py $SMOKE_FLAG --indexed --restart $MAX_Q
elif [ "$INDEXED" -eq 1 ]; then
  echo "==> Exp F indexed pass (search-only)"
  "$PY" scripts/exp_f_runtime.py $SMOKE_FLAG --indexed $RESTART $MAX_Q
else
  echo "==> Exp F runtime / workflow comparison (paper=$PAPER)"
  "$PY" scripts/exp_f_runtime.py $SMOKE_FLAG $RESTART $SKIP_EMB $MAX_Q
fi

if [ "$PLOT" -eq 1 ] || [ "$INDEXED" -eq 1 ]; then
  echo ""
  echo "==> Plotting Exp F figures"
  "$PY" "$PS/plot_exp_f_workflow.py"
fi

echo ""
if [ "$PAPER" -eq 0 ]; then
  echo "Artifacts: results/exp_f_smoke/"
else
  echo "Artifacts: results/exp_f/"
fi
echo "  metrics.json  metrics_indexed.json  workflow_comparison.tsv  summary.tsv"
echo "Paper section: paper/experiment_f.md"
