# Cross-cut X2 — Subspace coherence diagnostic (`subspace_coherence`)

**XQdrant change needed:** **None.** Pure offline NumPy on the corpus — runnable today, no server.

**What it proves:** a principled predictor of *when masking is safe*. If full-space and subspace
k-NN sets overlap strongly, masked traversal should keep recall; if they diverge, expect the
Option 1 recall collapse. Overlay this curve on the Option 1/2 recall curves to explain them.

## Run

```bash
python run_xcut2_coherence.py --dimensions 768 1536 --ratios 0.1 0.25 0.5 0.75 1.0
```

## Output

`experiments/<ts>__xcut2_subspace_coherence__subspace_coherence/`
- `results/xcut2_coherence_d{D}.csv` — overlap + Jaccard per ratio
- `plots/xcut2_coherence_d{D}.png|pdf`
- `manifest.json`
