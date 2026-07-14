# XQdrant masked-distance study — design docs

Internal writeups for the Options / cross-cuts implemented on branch
`option2-hybrid-layer-cutoff`. For **which experiment folders to cite**, see
[`../DB_systems/experiments/README.md`](../DB_systems/experiments/README.md).

| Doc | Topic | Decision |
|-----|--------|----------|
| [`option2.md`](./option2.md) | Hybrid `mask_from_layer` | **Keep** (navigability); e2e speedup still &lt; 1 |
| [`X1.md`](./X1.md) | Gather vs repack kernels | **Keep** microbench win; e2e still no speedup |
| [`X2.md`](./X2.md) | Subspace↔full k-NN coherence | **Keep** diagnostic (explains high-D / X3) |
| [`X3.md`](./X3.md) | `focus.verify` ± overfetch | **Keep** API + overfetch subplot; demote as latency/high-D fix |
| [`option4.md`](./option4.md) | `focus.alpha` blend | **Skip** as Option 2 enhancement |

Related plan / milestones: `docs/steps.md`, `docs/steps_vdbt.md`, `docs/working.md`.
Harness: `DB_systems/research/`.
