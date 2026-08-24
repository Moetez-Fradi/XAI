#!/usr/bin/env bash
#
# Download and unpack the SIFT1M corpus (TEXMEX) into DB_systems/data/sift/.
#
# Produces:
#   data/sift/sift_base.fvecs        1,000,000 x 128  (base vectors)
#   data/sift/sift_query.fvecs          10,000 x 128  (queries)
#   data/sift/sift_groundtruth.ivecs    10,000 x 100  (top-100 by L2)
#   data/sift/sift_learn.fvecs         100,000 x 128  (train, unused here)
#
# Usage:
#   ./fetch_sift.sh            # -> DB_systems/data/sift
#   ./fetch_sift.sh /some/dir  # custom destination
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/data/sift}"
URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

mkdir -p "$DEST"
cd "$DEST"

if [ -f sift_base.fvecs ] && [ -f sift_query.fvecs ] && [ -f sift_groundtruth.ivecs ]; then
  echo "SIFT1M already present in $DEST"
  ls -la
  exit 0
fi

echo "Downloading SIFT1M (~161 MB compressed) from:"
echo "  $URL"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o sift.tar.gz "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O sift.tar.gz "$URL"
else
  echo "ERROR: need curl or wget to download." >&2
  exit 1
fi

echo "Extracting ..."
# Tarball contains a top-level sift/ folder; strip it so files land directly in $DEST.
tar -xzf sift.tar.gz --strip-components=1
rm -f sift.tar.gz

echo "Done. Files in $DEST:"
ls -la
