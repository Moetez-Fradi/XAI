# Cross-cut X2 — Subspace coherence (`subspace_coherence`)

**XQdrant change:** **none** — offline NumPy. See `xqdrant_docs/X2.md`.

## Run

```bash
python run_xcut2_coherence.py --dataset sift1m --queries 500
python run_xcut2_coherence.py --dataset synthetic --dimensions 768 1536 \
  --num-vectors 50000 --queries 500
```

## Canonical results

| Corpus | Folder |
|--------|--------|
| SIFT | `experiments/2026-07-13_14-33-24__xcut2_subspace_coherence__subspace_coherence` |
| High-D | `experiments/2026-07-13_14-34-23__xcut2_subspace_coherence__subspace_coherence` |

Overlap grows with ratio but stays low on high-D until large `D_sub/D` — explains Option 2
plateau and X3 overfetch limits.
