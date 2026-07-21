# XQdrant masked-distance study — design docs

Internal writeups for mechanisms implemented on the XQdrant fork. The
**submission paper** ([`../deliverable/`](../deliverable/)) uses M1/M2/… names;
this folder and `DB_systems/research/` still use Option/X names.

## Paper ↔ research naming

| Paper | This folder / research dir | Decision |
|-------|----------------------------|----------|
| M2 | [`option2.md`](./option2.md) / `option2_hybrid_layer_cutoff` | **Keep** (navigability); e2e speedup still &lt; 1 |
| K1 | [`X1.md`](./X1.md) / `xcut1_gather_vs_repack` | **Keep** microbench win; e2e still no speedup |
| C1 | [`X2.md`](./X2.md) / `xcut2_subspace_coherence` | **Keep** diagnostic |
| V1 | [`X3.md`](./X3.md) / `xcut3_verify_pass` | **Keep** API + overfetch; demote as latency/high-D fix |
| M3 | [`option4.md`](./option4.md) / `option4_weighted_blend` | **Skip** as M2 enhancement |
| M1 | (baseline; see research `option1_naive_masked`) | Baseline only |

Related plan / milestones: [`../docs/`](../docs/).  
Harness: [`../DB_systems/research/`](../DB_systems/research/).  
Canonical figures for the paper: [`../deliverable/figures/`](../deliverable/figures/).
Experiment CSVs are regenerated locally under `DB_systems/experiments/` (gitignored).
