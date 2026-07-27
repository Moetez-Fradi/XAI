#!/usr/bin/env bash
# Create Bioinformatics/.venv with uv and install Python deps.
#
# Usage (from Bioinformatics/):
#   ./setup_env.sh
#   ./setup_env.sh --cuda          # Linux + NVIDIA: install CUDA torch wheels
#   ./setup_env.sh --skip-torch    # install requirements without torch (rare)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CUDA=0
SKIP_TORCH=0
for arg in "$@"; do
  case "$arg" in
    --cuda) CUDA=1 ;;
    --skip-torch) SKIP_TORCH=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

OS="$(uname -s)"
ARCH="$(uname -m)"
echo "==> OS=$OS arch=$ARCH"
echo "==> Creating .venv with uv (python 3.11 if available)"

if [ ! -d .venv ]; then
  uv venv --python 3.11 .venv 2>/dev/null || uv venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip tooling inside venv via uv"
uv pip install --python .venv/bin/python -U pip setuptools wheel

if [ "$SKIP_TORCH" -eq 0 ]; then
  if [ "$CUDA" -eq 1 ]; then
    if [ "$OS" != "Linux" ]; then
      echo "WARN: --cuda is intended for Linux+NVIDIA. Continuing anyway." >&2
    fi
    echo "==> Installing PyTorch (CUDA 12.1 index) then remaining requirements"
    uv pip install --python .venv/bin/python \
      torch torchvision \
      --index-url https://download.pytorch.org/whl/cu121
  else
    echo "==> Installing PyTorch from default index (CPU / macOS MPS wheels)"
  fi
fi

echo "==> Installing requirements.txt"
uv pip install --python .venv/bin/python -r requirements.txt

echo ""
echo "Environment ready."
echo "  activate:  source $ROOT/.venv/bin/activate"
echo "  downloads: ./run_downloads.sh"
echo "  tools:     ./fetch_tools_macos.sh   OR   ./fetch_tools_linux.sh"
