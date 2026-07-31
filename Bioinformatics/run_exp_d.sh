#!/usr/bin/env bash
# Experiment D — thermo/meso ortholog case study (full paper)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

export PATH="$ROOT/tools/mamba/envs/bio-tools/bin:$PATH"

if [ ! -x "$PY" ]; then
  echo "ERROR: missing .venv. Run ./setup_env.sh first." >&2
  exit 1
fi

SMOKE=0
RESTART=""
SKIP_OMA=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --restart) RESTART="--restart" ;;
    --skip-oma) SKIP_OMA=1 ;;
    *)
      echo "Usage: $0 [--smoke] [--restart] [--skip-oma]" >&2
      exit 1
      ;;
  esac
done

MODE=()
[ "$SMOKE" -eq 1 ] && MODE=(--smoke)

if [ "$SKIP_OMA" -eq 0 ] && [ "$SMOKE" -eq 0 ]; then
  echo "==> OMA thermophile ortholog lookup (all seeds)"
  "$PY" scripts/download_oma_thermo.py
  echo ""
fi

echo "==> Curate literature ortholog pairs"
"$PY" scripts/curate_exp_d_literature.py "${MODE[@]}"

echo ""
echo "==> Build thermo/meso pairs"
"$PY" scripts/prep_exp_d_pairs.py "${MODE[@]}"

echo ""
echo "==> Exp D thermo case study (requires XQdrant: ./start_xqdrant.sh --daemon)"
"$PY" scripts/exp_d_thermo.py "${MODE[@]}" $RESTART

echo ""
echo "Done → results/exp_d/ (or results/exp_d_smoke/)"
