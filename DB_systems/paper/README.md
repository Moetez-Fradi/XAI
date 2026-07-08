# XQdrant Paper — LaTeX Build

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Full camera-ready draft (~12 pages with figures) |
| `references.bib` | BibTeX bibliography |
| `../plots/*.pdf` | Benchmark figures (referenced via `\graphicspath`) |

## Compile

Requires the ACM `acmart` document class. On **BasicTeX**, install dependencies:

```bash
export PATH="/Library/TeX/texbin:$PATH"
sudo tlmgr install preprint   # provides balance.sty (no standalone "balance" package in TL2026)
```

If `newtx` is unavailable, the `newtxmath` warning is harmless (acmart falls back to default fonts).

```bash
cd DB_systems/paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or upload `paper/` + `plots/` to [Overleaf](https://www.overleaf.com) using the ACM SIGCONF template.

## Figures

The manuscript references:
- `plot1_latency_recall_d{768,1536}.pdf`
- `plot2_throughput_d768.pdf`
- `plot3_attribution_depth_d{768,1536}.pdf`
- `plot4_subspace_speedup_d768.pdf`

Regenerate plots from CSV without re-benchmarking:

```bash
cd DB_systems
python3 regenerate_plots.py
```

## Anonymous submission

`main.tex` uses `\documentclass[sigconf,anonymous,review]{acmart}`.
Remove `anonymous,review` and populate author blocks for camera-ready.
