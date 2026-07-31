#!/usr/bin/env bash
# Smoke Exp B v2: DSSP binned profiles → profile-level attribution validity.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

export PATH="$ROOT/tools/mamba/envs/bio-tools/bin:$PATH"

if [ ! -x "$PY" ]; then
  echo "ERROR: missing .venv. Run ./setup_env.sh first." >&2
  exit 1
fi

if [ ! -f results/exp_a_smoke/checkpoints/queries.jsonl ]; then
  echo "ERROR: need Exp A smoke results. Run ./run_exp_a_smoke.sh first." >&2
  exit 1
fi

echo "==> DSSP annotate + binned profiles (smoke)"
"$PY" scripts/annotate_dssp.py --smoke --from-exp-a --also-train 80 --store-profiles

echo "==> Exp B v2 attribution (smoke)"
"$PY" scripts/exp_b_v2_attribution.py --smoke --restart --n-perm 200

echo ""
echo "Smoke Exp B v2 done. Paper path:"
echo "  export PATH=\"$ROOT/tools/mamba/envs/bio-tools/bin:\$PATH\""
echo "  .venv/bin/python scripts/annotate_dssp.py --from-exp-a --also-train 3000 --store-profiles"
echo "  .venv/bin/python scripts/exp_b_v2_attribution.py"
echo ""
echo "Artifacts: results/exp_b_v2_smoke/ (smoke) or results/exp_b_v2/ (paper)"
echo "See paper/experiment_b.md and results/exp_b_v2/"
