# Option 4 — Weighted blend distance (`recall_speed_alpha`)

**XQdrant:** `focus.alpha` **implemented**. Score =
`alpha*s_full + (1-alpha)*s_focus` on focus layers. Not a speedup alone — pair with
`mask_from_layer`. See `xqdrant_docs/option4.md`.

**What it tests:** whether blending lets you mask more aggressively while keeping
subspace recall. (Harness uses subspace GT at a fixed `--ratio`.)

## Run

```bash
cd /mnt/data/xq/DB_systems && source .venv/bin/activate
cd research/option4_weighted_blend

python run_option4_alpha_sweep.py --mode http \
  --xqdrant-url http://127.0.0.1:6333 --qdrant-url http://127.0.0.1:6333 \
  --dataset synthetic --dimensions 768 1536 --num-vectors 50000 --queries 500 \
  --ratio 0.25 --alphas 0 0.25 0.5 0.75 1.0 --layers 0 1 2 \
  --ef-search 128 --k 10

python run_option4_alpha_sweep.py --mode http \
  --xqdrant-url http://127.0.0.1:6333 --qdrant-url http://127.0.0.1:6333 \
  --dataset sift1m --queries 500 \
  --ratio 0.25 --alphas 0 0.25 0.5 0.75 1.0 --layers 0 1 2 \
  --ef-search 128 --k 10
```

## Canonical results

| Corpus | Folder |
|--------|--------|
| High-D | `experiments/2026-07-14_12-31-59__option4_weighted_blend__recall_speed_alpha` |
| SIFT | `experiments/2026-07-14_12-37-16__option4_weighted_blend__recall_speed_alpha` |

## Keep / skip

**Skip as an Option 2 enhancement.** Best cell is `alpha=0` (pure hybrid masked layers);
raising alpha monotonically hurts subspace recall (SIFT layer=1: 0.84→0.08 as α→1).
Hard `mask_from_layer` alone is better. See `xqdrant_docs/option4.md` and
[`../../experiments/README.md`](../../experiments/README.md).
