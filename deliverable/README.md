# Short-track deliverable (ACM sigconf)

**Status (2026-07-21):** ready to skim for short-track submission.

## What is frozen here
- Unified-host HTTP evals with **T=5** mean±std for attribution (A/C/D), SIFT M1/M2, and SIFT coherence (C1).
- Figures under `figures/` refreshed from those runs (M3 panel restored from the canonical SIFT α×cutoff sweep).
- `main.tex` numbers aligned with those results (~1.8× attribution speedup; updated M1/M2 recall ranges).

## Build
Requires a TeX Live with `acmart` (TinyTeX works):

```bash
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"   # if using TinyTeX
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Not required for short track
Items in `DB_systems/step 2/steps.md` beyond variance (formal M2 complexity writeup, second dataset, filtering/quantization studies, case study) — those are full-track expansions.
