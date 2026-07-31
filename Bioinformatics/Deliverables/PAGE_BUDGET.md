# PSB 2027 page budget (target: ≤12 body pages)

**Not counted toward 12:** cover letter, title/author block, references.

Run after edits:

```bash
cd Bioinformatics/Deliverables
make figures && make && make pagecount
```

## Figures in final draft

1. Pipeline schematic
2. Exp A recall bars
3. Exp B violin + label-perm (2-panel)
4. Exp B top-N ablation
5. Exp C ROC
6. Exp D scatter + forest + PyMOL (2-panel)
7. Exp E site enrichment + attribution/PyMOL (2 figures)
8. Exp F workflow timing
9. Layer/pooling ablation

## If over 12 body pages

1. Move layer/pooling table to supplement; keep figure only
2. Combine Exp D forest into supplement
3. Shrink PyMOL panels or move one to supplement URL

## Before submission

- [ ] Fill author block + `cover_letter.tex` email
- [ ] Rename PDF to `lastname.pdf` per PSB rules
- [ ] Verify page count ≤12 (body only)
- [ ] LLM disclosure in cover letter (draft included)
