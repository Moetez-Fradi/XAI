#!/usr/bin/env bash
# Install all LaTeX packages required to build paper/main.tex on macOS BasicTeX.
set -euo pipefail

export PATH="/Library/TeX/texbin:$PATH"

if ! command -v tlmgr &>/dev/null; then
  echo "ERROR: tlmgr not found. Install BasicTeX first:"
  echo "  brew install --cask basictex"
  echo "  export PATH=\"/Library/TeX/texbin:\$PATH\""
  exit 1
fi

echo "Updating tlmgr..."
sudo tlmgr update --self

echo "Installing ACM + paper dependencies..."
sudo tlmgr install \
  acmart \
  amsmath amsfonts amssymb \
  booktabs graphicx subcaption xcolor \
  algorithms algorithmicx \
  libertine inconsolata newtx \
  comment hyperref hyperxmp xstring \
  natbib microtype geometry caption float fancyhdr \
  environ totpages refcount everyshi trimspaces \
  ncctools manyfoot nccfoots \
  etoolbox kvoptions kvsetkeys pdftexcmds \
  collection-fontsrecommended collection-latexrecommended preprint

# Optional: uncomment if packages are still missing (~150MB extra)
# sudo tlmgr install scheme-medium

echo ""
echo "Done. Compile with:"
echo "  cd $(dirname "$0")"
echo "  pdflatex main && bibtex main && pdflatex main && pdflatex main"
