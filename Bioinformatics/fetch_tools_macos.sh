#!/usr/bin/env bash
# Install bioinformatics CLI tools on macOS (Homebrew + optional micromamba).
#
# Tools: dssp (mkdssp), foldseek, blast+, tmalign (via conda-forge when possible),
#        freesasa. PyMOL: brew or conda (open-source).
#
# Usage:
#   ./fetch_tools_macos.sh
#   ./fetch_tools_macos.sh --brew-only
#   ./fetch_tools_macos.sh --conda-only
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$ROOT/tools"
mkdir -p "$TOOLS"
cd "$ROOT"

BREW_ONLY=0
CONDA_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --brew-only) BREW_ONLY=1 ;;
    --conda-only) CONDA_ONLY=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

have() { command -v "$1" >/dev/null 2>&1; }

echo "==> macOS tool bootstrap → $TOOLS"
echo "    Prefer Apple Silicon Homebrew for blast/dssp; Foldseek/TM-align via conda-forge."

if [ "$CONDA_ONLY" -eq 0 ]; then
  if ! have brew; then
    echo "ERROR: Homebrew not found. Install from https://brew.sh" >&2
    exit 1
  fi
  echo "==> brew: dssp, blast, freeglut deps as available"
  brew list dssp >/dev/null 2>&1 || brew install brewsci/bio/dssp || brew install dssp || true
  brew list blast >/dev/null 2>&1 || brew install blast || true
  # FreeSASA often via pip later; brew formula may not exist
fi

if [ "$BREW_ONLY" -eq 1 ]; then
  echo "Brew-only done. Foldseek / TM-align / PyMOL: re-run without --brew-only."
  exit 0
fi

# micromamba in tools/ if no conda
ARCH="$(uname -m)"
case "$ARCH" in
  arm64) MM_ARCH=osx-arm64 ;;
  x86_64) MM_ARCH=osx-64 ;;
  *) echo "Unsupported macOS arch: $ARCH" >&2; exit 1 ;;
esac

if ! have conda && ! have micromamba && ! have mamba; then
  echo "==> Installing micromamba ($MM_ARCH) into $TOOLS"
  curl -LfsS "https://micro.mamba.pm/api/micromamba/${MM_ARCH}/latest" \
    | tar -xvj -C "$TOOLS" bin/micromamba
  export MAMBA_ROOT_PREFIX="$TOOLS/mamba"
  eval "$("$TOOLS/bin/micromamba" shell hook -s bash)"
  "$TOOLS/bin/micromamba" create -y -n bio-tools -c conda-forge -c bioconda \
    foldseek tmalign freesasa openmm 2>/dev/null \
    || "$TOOLS/bin/micromamba" create -y -n bio-tools -c conda-forge -c bioconda foldseek
  echo "Activate tools env:"
  echo "  eval \"\$($TOOLS/bin/micromamba shell hook -s bash)\""
  echo "  micromamba activate bio-tools"
else
  echo "==> Using existing conda/mamba/micromamba"
  SOLVER=conda
  have mamba && SOLVER=mamba
  have micromamba && SOLVER=micromamba
  $SOLVER create -y -n bio-tools -c conda-forge -c bioconda foldseek || true
  $SOLVER install -y -n bio-tools -c conda-forge -c bioconda tmalign freesasa || true
fi

# PyMOL: optional large install
if have brew; then
  echo "==> Optional: brew install --cask pymol  (GUI) or conda install -c conda-forge pymol-open-source"
fi

cat > "$TOOLS/env_macos.sh" <<EOF
# Source after fetch_tools_macos.sh
export PATH="$TOOLS/bin:\$PATH"
# If micromamba env exists:
# eval "\$($TOOLS/bin/micromamba shell hook -s bash)"
# micromamba activate bio-tools
EOF

echo "macOS tools bootstrap finished. See $TOOLS/env_macos.sh"
