# Research planning notes

Internal planning docs for the XQdrant study. The **submission paper** lives in
[`../deliverable/`](../deliverable/) and uses the **M1/M2/…** names below.
Harness scripts still use the older Option/X directory names.

## Paper ↔ research naming

| Paper | Research dir / old name | Role |
|-------|-------------------------|------|
| M1 | Option 1 / `option1_naive_masked` | Naive all-layer mask |
| M2 | Option 2 / `option2_hybrid_layer_cutoff` | Hybrid `mask_from_layer` |
| M3 | Option 4 / `option4_weighted_blend` | α-blend (skip) |
| K1 | X1 / `xcut1_gather_vs_repack` | Gather vs repack kernel |
| C1 | X2 / `xcut2_subspace_coherence` | Subspace↔full coherence |
| V1 | X3 / `xcut3_verify_pass` | Verify ± overfetch |
| — | X4 / `xcut4_divergence` | Visited-set divergence (full-paper) |

## Files here

| File | Contents |
|------|----------|
| [`steps.md`](./steps.md) | Option/X execution plan |
| [`working.md`](./working.md) | API / milestone notes |

Full-track expansion plan (M-naming): [`../DB_systems/full_paper_plan/steps.md`](../DB_systems/full_paper_plan/steps.md).  
Mechanism decision docs: [`../xqdrant_docs/`](../xqdrant_docs/).  
Bench harness: [`../DB_systems/`](../DB_systems/).
