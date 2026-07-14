# XQdrant Paper — LaTeX Build

## Status

`main.tex` is a **full first draft** covering:
1. In-database attribution (`with_dims_explained`)
2. Masked-distance HNSW (Options 1/2/4, cross-cuts X1–X3)

Compiled PDF: **`main.pdf`** (article class, ~11 pages with figures).

Canonical experiment folders for numbers/plots:
[`../experiments/README.md`](../experiments/README.md).

## Compile (this machine)

The host TeX tree is incomplete (missing `acmart` deps / `hyperref` extras).
The draft uses a portable `article` preamble so it builds locally:

```bash
cd DB_systems/paper
export TMPDIR=$PWD/.tmp   # avoid /tmp quota issues
mkdir -p "$TMPDIR"
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode main
```

Figures are staged under `figures/` from `../experiments/`.

## ACM / Overleaf camera-ready

Local `acmart.cls` was generated from CTAN sources in `acmart/`.
On a full TeX Live (or Overleaf ACM template):

1. Restore the `acmart` preamble (see git history / `install_tex_deps.sh`).
2. Install `texlive-publishers` (provides `acmart`) and friends.
3. Point `\graphicspath` at `figures/`.

## Files

| Path | Purpose |
|------|---------|
| `main.tex` | Manuscript body |
| `main.pdf` | Compiled first draft |
| `references.bib` | BibTeX (for Overleaf/acmart builds) |
| `figures/` | Staged plots |
| `acmart/` | Upstream ACM class sources |
