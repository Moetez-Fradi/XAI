#!/usr/bin/env bash
#
# Download and unpack the GIST1M corpus (TEXMEX) into data/gist/.
#
# Produces:
#   data/gist/gist_base.fvecs        1,000,000 x 960
#   data/gist/gist_query.fvecs          1,000 x 960
#   data/gist/gist_groundtruth.ivecs    1,000 x 100
#   data/gist/gist_learn.fvecs         500,000 x 960
#
# Usage:
#   ./fetch_gist.sh            # -> VLDB 2027/data/gist
#   ./fetch_gist.sh /some/dir  # custom destination
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/data/gist}"
URL="ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"

mkdir -p "$DEST"
cd "$DEST"

if [ -f gist_base.fvecs ] && [ -f gist_query.fvecs ]; then
  echo "GIST1M already present in $DEST"
  ls -la
  exit 0
fi

echo "Downloading GIST1M from:"
echo "  $URL"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o gist.tar.gz "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O gist.tar.gz "$URL"
else
  echo "ERROR: need curl or wget to download." >&2
  exit 1
fi

echo "Extracting ..."
tar -xzf gist.tar.gz --strip-components=1
rm -f gist.tar.gz

echo "Done. Files in $DEST:"
ls -la
