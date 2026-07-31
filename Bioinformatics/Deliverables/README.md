# PSB 2027 Deliverables — XQdrant Paper Draft

Draft manuscript for Pacific Symposium on Biocomputing 2027.  
**Page limit:** 12 pages **excluding** cover letter, title/author page, and references.

## Quick build

```bash
cd Bioinformatics/Deliverables
make clean && make
make pagecount    # estimate body page count
```

Output: `xqdrant_psb2027.pdf` (cover letter + manuscript + references).

Requirements: `pdflatex`, `bibtex` (TeX Live). On minimal installs (Arch `texlive-basic` + `texlive-latex`), local stubs for `chapterbib.sty` and `infwarerr.sty` are copied automatically. For full compatibility, install `texlive-latexextra`.

## Contents

| Path | Purpose |
|------|---------|
| `xqdrant_psb2027.tex` | Main manuscript draft (Experiments A–D) |
| `cover_letter.tex` | PSB-required cover page (first page of PDF) |
| `xqdrant.bib` | Starter bibliography |
| `template/` | Official PSB `ws-procs11x85` LaTeX class (from psb.stanford.edu) |
| `figures/` | Copied from `plot_scripts/outputs/` |
| `PAGE_BUDGET.md` | Section length / page planning |
| `Makefile` | Build + page count |

## Before submission

1. Replace placeholder author name, email, affiliation in `xqdrant_psb2027.tex` and `cover_letter.tex`.
2. Update LLM disclosure if needed.
3. Run `make pagecount` — body (excl. cover, title block, references) should be ≤ 12 pages.
4. Rename PDF to `lastname.pdf` per PSB instructions.
5. Submit via [PSB paper management system](https://psb.stanford.edu/psb-online/psb-submit/) (Wufoo).

## Regenerate figures

```bash
cd Bioinformatics
./plot_scripts/run_all.sh
cp plot_scripts/outputs/*.pdf plot_scripts/outputs/pymol/*.png Deliverables/figures/
make -C Deliverables
```

## Status (Experiments A–D)

| Exp | In draft | Figures |
|-----|----------|---------|
| A | ✅ | `exp_a_retrieval_recall.pdf` |
| B v3 | ✅ | violin + label-perm |
| C | ✅ | `exp_c_roc.pdf` |
| D | ✅ | scatter, forest, PyMOL PNGs |
| E, F | Mentioned as planned | — |

## Official template

Downloaded from: http://psb.stanford.edu/psb-online/psb-submit/psb11x85_2e.zip
