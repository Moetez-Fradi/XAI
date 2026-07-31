#!/usr/bin/env bash
# Layer + pooling ablation (steps.md §4) — embed subset + score attribution fold-gap.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
ACFG="configs/exp_ablation.yaml"

DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-2}"
RESTART=""
SCORE_ONLY=0
EMBED_ONLY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --restart) RESTART="--force" ;;
    --device)
      shift
      DEVICE="$1"
      ;;
    --batch-size)
      shift
      BATCH_SIZE="$1"
      ;;
    --embed-only)
      shift
      EMBED_ONLY="$1"
      ;;
    --score-only) SCORE_ONLY=1 ;;
    *)
      echo "Usage: $0 [--restart] [--device cuda|cpu] [--batch-size N] [--embed-only CONFIG] [--score-only]" >&2
      exit 1
      ;;
  esac
  shift
done

if [ ! -x "$PY" ]; then
  echo "ERROR: run ./setup_env.sh first" >&2
  exit 1
fi

echo "==> Curate ablation chain set"
if [ -z "$RESTART" ]; then
  "$PY" scripts/prep_exp_ablation.py
else
  "$PY" scripts/prep_exp_ablation.py --force
fi

CHAINS="data/processed/exp_ablation/chain_ids.txt"
N=$(wc -l < "$CHAINS")
if [ ! -f embeddings/ablation/layer33_mean/vectors.npy ] || [ -n "$RESTART" ]; then
  echo "==> Baseline layer33_mean from full corpus (extract subset)"
  "$PY" scripts/extract_ablation_embeddings.py
else
  echo "==> Skip extract layer33_mean (already present)"
fi

echo "==> Embed remaining configs ($N chains, device=$DEVICE, batch=$BATCH_SIZE)"

embed_one() {
  local name="$1" layer="$2" pooling="$3" outdir="$4"
  if [ -f "$outdir/vectors.npy" ] && [ -z "$RESTART" ]; then
    echo "  skip $name (exists)"
    return 0
  fi
  echo "  embed $name → $outdir"
  "$PY" scripts/embed_esm2.py \
    --chains-file "$CHAINS" \
    --outdir "$outdir" \
    --layer "$layer" \
    --pooling "$pooling" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE"
}

if [ "${SCORE_ONLY:-0}" -eq 0 ]; then
  if [ -z "$EMBED_ONLY" ] || [ "$EMBED_ONLY" = "layer16_mean" ]; then
    embed_one layer16_mean 16 mean embeddings/ablation/layer16_mean
  fi
  if [ -z "$EMBED_ONLY" ] || [ "$EMBED_ONLY" = "layer8_mean" ]; then
    embed_one layer8_mean 8 mean embeddings/ablation/layer8_mean
  fi
  if [ -z "$EMBED_ONLY" ] || [ "$EMBED_ONLY" = "layer33_cls" ]; then
    embed_one layer33_cls last cls embeddings/ablation/layer33_cls
  fi
fi

echo ""
echo "==> Score attribution validity per config"
"$PY" scripts/exp_ablation_attribution.py

echo ""
echo "==> Plot ablation comparison"
"$PY" plot_scripts/plot_exp_ablation.py

echo ""
echo "Done → results/exp_ablation/  plot_scripts/outputs/exp_ablation_*.png"
