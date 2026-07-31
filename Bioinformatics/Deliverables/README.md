# PSB 2027 Deliverables — XQdrant Paper Draft

Draft manuscript for Pacific Symposium on Biocomputing 2027.  
**Page limit:** 12 pages **excluding** cover letter, title/author page, and references.

## Quick build

```bash
cd Bioinformatics/Deliverables
make figures && make
make pagecount
```

Output: `xqdrant_psb2027.pdf` (cover letter + manuscript + references).

Requirements: `pdflatex`, `bibtex` (TeX Live). Local stubs for `chapterbib.sty` and `infwarerr.sty` included for minimal TeX Live.

## Contents

| Path | Purpose |
|------|---------|
| `xqdrant_psb2027.tex` | Main manuscript (Experiments A–F + ablations) |
| `cover_letter.tex` | PSB-required cover page |
| `xqdrant.bib` | Bibliography |
| `figures/` | Paper figures |
| `PAGE_BUDGET.md` | Page planning |
| `Makefile` | Build |

## Before submission

1. Replace placeholder author name, email, affiliation in `xqdrant_psb2027.tex` and `cover_letter.tex`.
2. Run `make pagecount` — body ≤ 12 pages.
3. Rename PDF to `fradi.pdf` (or `lastname.pdf`) per PSB instructions.
4. Optional: post same PDF on [bioRxiv](https://www.biorxiv.org/submit-a-manuscript) — see `biorxiv_submission.txt`.
5. Submit via [PSB paper management](https://psb.stanford.edu/psb-online/psb-submit/).

## Regenerate figures

```bash
cd Bioinformatics
./plot_scripts/run_all.sh
make -C Deliverables figures
make -C Deliverables
```

## Status (Experiments A–F)

| Exp | In draft | Key result |
|-----|----------|------------|
| A | ✅ | R@1=0.898, mAP=0.896 |
| B v3 | ✅ | MW p≈10⁻⁵⁸; label-perm p=0.0002 |
| C | ✅ | AUROC=0.63 |
| D | ✅ | ion-pair ρ=−0.30, p=0.052 |
| E | ✅ | site enrichment + 1UWL_B PyMOL panel |
| F | ✅ | indexed 0.07s vs Foldseek+TM 4.17s |
| Ablations | ✅ | only layer33_mean fold-gap + |

## Official template

From: http://psb.stanford.edu/psb-online/psb-submit/psb11x85_2e.zip
