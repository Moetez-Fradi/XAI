#!/usr/bin/env bash
# Install bioinformatics CLI tools on Linux (apt where useful + micromamba/bioconda).
#
# Tools: dssp, foldseek, blast+, tmalign, freesasa, optional pymol-open-source.
#
# Usage:
#   ./fetch_tools_linux.sh
#   ./fetch_tools_linux.sh --apt-only
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$ROOT/tools"
mkdir -p "$TOOLS"
cd "$ROOT"

APT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --apt-only) APT_ONLY=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

have() { command -v "$1" >/dev/null 2>&1; }

echo "==> Linux tool bootstrap → $TOOLS"

if have apt-get; then
  echo "==> apt packages (may need sudo)"
  sudo apt-get update -y
  sudo apt-get install -y \
    ncbi-blast+ \
    dssp \
    curl \
    wget \
    tar \
    bzip2 \
    || true
else
  echo "WARN: apt-get not found; relying on conda-forge/bioconda only."
fi

if [ "$APT_ONLY" -eq 1 ]; then
  echo "apt-only done."
  exit 0
fi

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) MM_ARCH=linux-64 ;;
  aarch64|arm64) MM_ARCH=linux-aarch64 ;;
  *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac

if ! have conda && ! have micromamba && ! have mamba; then
  echo "==> Installing micromamba ($MM_ARCH) into $TOOLS"
  curl -LfsS "https://micro.mamba.pm/api/micromamba/${MM_ARCH}/latest" \
    | tar -xvj -C "$TOOLS" bin/micromamba
  export MAMBA_ROOT_PREFIX="$TOOLS/mamba"
  "$TOOLS/bin/micromamba" create -y -n bio-tools -c conda-forge -c bioconda \
    foldseek tmalign freesasa pymol-open-source || \
  "$TOOLS/bin/micromamba" create -y -n bio-tools -c conda-forge -c bioconda foldseek
  echo "Activate:"
  echo "  eval \"\$($TOOLS/bin/micromamba shell hook -s bash)\""
  echo "  micromamba activate bio-tools"
else
  SOLVER=conda
  have mamba && SOLVER=mamba
  have micromamba && SOLVER=micromamba
  $SOLVER create -y -n bio-tools -c conda-forge -c bioconda foldseek tmalign freesasa || true
fi

cat > "$TOOLS/env_linux.sh" <<EOF
# Source after fetch_tools_linux.sh
export PATH="$TOOLS/bin:\$PATH"
# eval "\$($TOOLS/bin/micromamba shell hook -s bash)"
# micromamba activate bio-tools
EOF

echo "Linux tools bootstrap finished. See $TOOLS/env_linux.sh"
echo "For full-corpus ESM2 embedding, prefer a CUDA GPU machine (./setup_env.sh --cuda)."
