# XQdrant masked-distance study — design docs

Internal writeups for mechanisms on the XQdrant fork. The **submission paper**
([`../deliverable/`](../deliverable/)) and the harness entry point
([`../DB_systems/README.md`](../DB_systems/README.md)) use **M1 / M2 / …** names.
This folder and `DB_systems/research/` directory names still use Option/X labels.

## Paper ↔ this folder

| Paper | Doc | Research dir | Decision |
|-------|-----|--------------|----------|
| M1 | (baseline; see research README) | `option1_naive_masked` | **Keep** (full-space recall = coherence) |
| M2 | [`option2.md`](./option2.md) | `option2_hybrid_layer_cutoff` | Optional (≤ +0.04 over M1); no speedup |
| M3 | [`option4.md`](./option4.md) | `option4_weighted_blend` | **Skip** |
| K1 | [`X1.md`](./X1.md) | `xcut1_gather_vs_repack` | **Keep** micro; masked still slower than plain |
| C1 | [`X2.md`](./X2.md) | `xcut2_subspace_coherence` | **Keep** (equals full-space recall of masked search) |
| V1 | [`X3.md`](./X3.md) | `xcut3_verify_pass` | **Keep** with overfetch (+0.2 ms in-engine) |

Harness runners (paper order): [`../DB_systems/research/README.md`](../DB_systems/research/README.md).  
Planning notes: [`../docs/`](../docs/).  
Canonical figures: [`../deliverable/figures/`](../deliverable/figures/).  
Local CSVs: `DB_systems/experiments/` (gitignored; see that folder’s README).
