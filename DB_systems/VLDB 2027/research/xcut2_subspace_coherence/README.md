# C1 — Subspace coherence (`xcut2_subspace_coherence`)

**Paper:** C1 · §2.3 definition · §5.6 results · Table 2: *keep* diagnostic  
**XQdrant change:** none (offline NumPy)  
**Folder name (legacy):** `xcut2_subspace_coherence`  
**Design doc:** [`../../../xqdrant_docs/X2.md`](../../../xqdrant_docs/X2.md)

**Metric:** mean `|N_full_k ∩ N_F_k| / k` (exact top-*k* overlap, full vs focus subspace).

**What it proves:** high-D **M2** recall plateaus and **V1** overfetch limits are
properties of the data/focus set, not of the traversal algorithm.

## Run

```bash
python run_xcut2_coherence.py --dataset sift1m --queries 500 --trials 5
python run_xcut2_coherence.py --dataset synthetic --dimensions 768 1536 \
  --num-vectors 50000 --queries 500 --trials 5
```

## Canonical results

| Corpus | Folder under `DB_systems/experiments/` |
|--------|----------------------------------------|
| SIFT | `2026-07-13_14-33-24__xcut2_subspace_coherence__subspace_coherence` |
| High-D | `2026-07-13_14-34-23__xcut2_subspace_coherence__subspace_coherence` |

See also: [`../README.md`](../README.md).
