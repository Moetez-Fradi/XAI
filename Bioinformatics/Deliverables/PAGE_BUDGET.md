# PSB 2027 page budget (target: ≤12 body pages)

**Not counted toward 12:** cover letter, title/author block, references.

## Draft estimate (first build: **9 pages total**)

| PDF page | Content | Counted in 12-page limit? |
|----------|---------|---------------------------|
| 1 | Cover letter | No |
| 2 | Title, authors, abstract, keywords | No (title page) |
| 3–7 | Introduction → Results (through Exp D scatter) | **Yes (~5 pages)** |
| 8 | Forest + PyMOL + References start | Refs: No; figures: Yes |
| 9 | References (end) | No |

**Estimated body pages (toward 12): ~5–6** — substantial room for Methods detail, pipeline figure, Exp B ablation, and Exp E/F stubs before hitting the limit.

## Figures in draft (8 panels / 5 figure environments)

1. Exp A recall bars
2. Exp B violin + label-perm (2-panel)
3. Exp C ROC
4. Exp D scatter (4-panel)
5. Exp D forest CI
6. Exp D PyMOL RSA + charged (2-panel)

## Optional additions (watch page limit)

- Pipeline schematic (0.5 page) — `steps.md` Fig 6
- Exp B ablation top-N (`exp_b_v3_ablation_topn.pdf`) — 0.25 page
- Exp D source breakdown — supplement or trim
- Exp A full comparison table in appendix URL

## If over 12 pages

1. Move ablation details to supplement URL
2. Combine Exp B/C into one subsection with smaller figures
3. Shorten Methods (move permutation details to supplement)
4. Use `wsdraft` class option for draft only (switch off for submission)

## If under 10 pages

1. Expand Methods (DSSP binning, pair construction for Exp D)
2. Add Exp B negative-control paragraph (saturated diff-fold)
3. Add Discussion on engineering workflows
4. Add pipeline figure

Run `make pagecount` after edits to verify.
