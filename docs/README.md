# Research planning notes

Internal planning for the XQdrant study. **Start with the paper and harness** if
you are reproducing results:

1. Paper: [`../deliverable/`](../deliverable/)
2. Harness map (paper → folders): [`../DB_systems/README.md`](../DB_systems/README.md)
3. Mechanism design writeups: [`../xqdrant_docs/`](../xqdrant_docs/)

This `docs/` folder is milestone / API scratch space — not the artifact index.

## Paper ↔ research naming

| Paper | Research dir / old name | Role |
|-------|-------------------------|------|
| M1 | Option 1 / `option1_naive_masked` | Naive all-layer mask |
| M2 | Option 2 / `option2_hybrid_layer_cutoff` | Hybrid `mask_from_layer` |
| M3 | Option 4 / `option4_weighted_blend` | α-blend (**skip**) |
| K1 | X1 / `xcut1_gather_vs_repack` | Gather vs repack kernel |
| C1 | X2 / `xcut2_subspace_coherence` | Subspace↔full coherence |
| V1 | X3 / `xcut3_verify_pass` | Verify ± overfetch |
| — | X4 / `xcut4_divergence` | Visited-set divergence (full-paper only) |

## Files here

| File | Contents |
|------|----------|
| [`steps.md`](./steps.md) | Option/X execution plan (historical) |
| [`working.md`](./working.md) | API / milestone notes |

Full-track expansion ideas: [`../DB_systems/full_paper_plan/steps.md`](../DB_systems/full_paper_plan/steps.md).
