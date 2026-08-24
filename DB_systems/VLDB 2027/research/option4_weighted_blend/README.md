# M3 — Weighted blend (`option4_weighted_blend`)

**Paper:** M3 · §4.4 design · §5.8 results · Table 2: **skip**  
**API:** `focus.alpha` ∈ [0, 1] with `masked=true`  
  (`score = α·s_full + (1−α)·s_focus` on masked layers)  
**Folder name (legacy):** `option4_weighted_blend`  
**Design doc:** [`../../../xqdrant_docs/option4.md`](../../../xqdrant_docs/option4.md)

**What it proves:** on SIFT, recall is maximized at **α=0** for every cutoff — i.e.
M3 collapses to **M2**. Raising α monotonically hurts subspace recall. Blending
computes both distances, so it is not a speedup either. Do not ship.

## Run

```bash
python run_option4_alpha_sweep.py --mode http \
  --xqdrant-url http://127.0.0.1:6333 --qdrant-url http://127.0.0.1:6333 \
  --dataset sift1m --queries 500 --trials 5 \
  --ratio 0.25 --alphas 0 0.25 0.5 0.75 1.0 --layers 0 1 2 \
  --ef-search 128 --k 10
```

## Canonical results

| Corpus | Folder under `DB_systems/experiments/` |
|--------|----------------------------------------|
| High-D | `2026-07-14_12-31-59__option4_weighted_blend__recall_speed_alpha` |
| SIFT | `2026-07-14_12-37-16__option4_weighted_blend__recall_speed_alpha` |

See also: [`../README.md`](../README.md).
