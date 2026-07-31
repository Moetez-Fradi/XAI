#!/usr/bin/env bash
# Exp B v3 — improved attribution validity (does not touch v1/v2 results).
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
BACKFILL=0
RESTART=""
N_PERM=""

while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) PAPER=0 ;;
    --backfill-profiles) BACKFILL=1 ;;
    --restart) RESTART="--restart" ;;
    --n-perm)
      shift
      N_PERM="--n-perm $1"
      ;;
    *)
      echo "Usage: $0 [--smoke] [--backfill-profiles] [--restart] [--n-perm N]" >&2
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
  echo "==> DSSP + profiles (smoke)"
  "$PY" scripts/annotate_dssp.py --smoke --from-exp-a --also-train 80 --store-profiles
  echo "==> Exp B v3 (smoke)"
  "$PY" scripts/exp_b_v3_attribution.py --smoke $RESTART ${N_PERM:---n-perm 200}
  echo "Done: results/exp_b_v3_smoke/"
  exit 0
fi

if [ ! -f results/exp_a/checkpoints/queries.jsonl ]; then
  echo "ERROR: need Exp A paper results." >&2
  exit 1
fi

if [ "$BACKFILL" -eq 1 ]; then
  echo "==> Backfill binned profiles (Exp A chains + extra train for map)"
  # Uses exp_b.yaml neighbor count; v3 reads top-10 from Exp A jsonl directly.
  "$PY" scripts/annotate_dssp.py --from-exp-a --max-neighbors 10 --store-profiles --also-train 8000
fi

echo "==> Exp B v3 attribution (paper)"
"$PY" scripts/exp_b_v3_attribution.py $RESTART $N_PERM

echo ""
echo "Paper run complete. Artifacts: results/exp_b_v3/"
echo "Compare: results/exp_b_v3/v2_vs_v3_comparison.tsv"
echo ""
echo "Optional before re-run:"
echo "  ./run_exp_b_v3.sh --backfill-profiles   # DSSP backfill + more map chains"
