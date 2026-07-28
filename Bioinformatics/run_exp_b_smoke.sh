#!/usr/bin/env bash
# Smoke Exp B: DSSP annotate (Exp A smoke chains + small train) → attribution test.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

# Ensure mkdssp on PATH for BioPython
export PATH="$ROOT/tools/mamba/envs/bio-tools/bin:$PATH"

if [ ! -f results/exp_a_smoke/checkpoints/queries.jsonl ]; then
  echo "ERROR: need Exp A smoke results. Run ./run_exp_a_smoke.sh first." >&2
  exit 1
fi

echo "==> DSSP annotate (smoke)"
"$PY" scripts/annotate_dssp.py --smoke --from-exp-a --also-train 80 --restart

echo "==> Exp B attribution (smoke)"
"$PY" scripts/exp_b_attribution.py --smoke --restart --n-perm 200

echo ""
echo "Smoke Exp B done. Paper:"
echo "  export PATH=\"$ROOT/tools/mamba/envs/bio-tools/bin:\$PATH\""
echo "  .venv/bin/python scripts/annotate_dssp.py --from-exp-a --also-train 3000"
echo "  .venv/bin/python scripts/exp_b_attribution.py"
