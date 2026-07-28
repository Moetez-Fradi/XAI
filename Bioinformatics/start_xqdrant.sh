#!/usr/bin/env bash
# Start the local XQdrant fork for Bioinformatics indexing / experiments.
#
# Usage (from Bioinformatics/):
#   ./start_xqdrant.sh              # foreground
#   ./start_xqdrant.sh --daemon     # background → data/xqdrant_storage/xqdrant.log
#   ./start_xqdrant.sh --stop
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BIN_DEFAULT="$ROOT/../XQdrant/target/release/xqdrant"
# allow override
BIN="${XQDRANT_BIN:-}"
STORAGE="${XQDRANT_STORAGE:-$ROOT/data/xqdrant_storage}"
HTTP_PORT="${XQDRANT_HTTP_PORT:-6333}"
GRPC_PORT="${XQDRANT_GRPC_PORT:-6334}"
PIDFILE="$STORAGE/xqdrant.pid"
LOGFILE="$STORAGE/xqdrant.log"

DAEMON=0
STOP=0
for arg in "$@"; do
  case "$arg" in
    --daemon) DAEMON=1 ;;
    --stop) STOP=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

if [ -z "$BIN" ]; then
  if [ -x "$BIN_DEFAULT" ]; then
    BIN="$BIN_DEFAULT"
  elif command -v xqdrant >/dev/null 2>&1; then
    BIN="$(command -v xqdrant)"
  else
    echo "ERROR: xqdrant binary not found at $BIN_DEFAULT" >&2
    echo "Build it: (cd ../XQdrant && cargo build --release --bin xqdrant)" >&2
    exit 1
  fi
fi

mkdir -p "$STORAGE"

if [ "$STOP" -eq 1 ]; then
  if [ -f "$PIDFILE" ]; then
    pid="$(cat "$PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
      echo "Stopped xqdrant pid=$pid"
    else
      echo "No running process for pidfile ($pid)"
    fi
    rm -f "$PIDFILE"
  else
    echo "No pidfile at $PIDFILE"
  fi
  exit 0
fi

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Already running pid=$(cat "$PIDFILE") — storage=$STORAGE"
  echo "  health: curl -s http://127.0.0.1:${HTTP_PORT}/readyz"
  exit 0
fi

export QDRANT__STORAGE__STORAGE_PATH="$STORAGE"
export QDRANT__STORAGE__SNAPSHOTS_PATH="$STORAGE/snapshots"
export QDRANT__SERVICE__HTTP_PORT="$HTTP_PORT"
export QDRANT__SERVICE__GRPC_PORT="$GRPC_PORT"

echo "==> Starting $BIN"
echo "    storage: $STORAGE"
echo "    http:    http://127.0.0.1:${HTTP_PORT}"

# Run from XQdrant repo root so default config/static resolution works
XQ_ROOT="$(cd "$ROOT/../XQdrant" && pwd)"
if [ ! -f "$XQ_ROOT/Cargo.toml" ]; then
  XQ_ROOT="$(cd "$(dirname "$BIN")/../.." && pwd)"
fi

if [ "$DAEMON" -eq 1 ]; then
  (
    cd "$XQ_ROOT"
    nohup "$BIN" --disable-telemetry >"$LOGFILE" 2>&1 &
    echo $! >"$PIDFILE"
  )
  sleep 1
  if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Daemon up pid=$(cat "$PIDFILE") log=$LOGFILE"
    echo "Next:"
    echo "  .venv/bin/python scripts/index_xqdrant.py --smoke"
    echo "  .venv/bin/python scripts/index_xqdrant.py"
  else
    echo "ERROR: failed to start — see $LOGFILE" >&2
    exit 1
  fi
else
  cd "$XQ_ROOT"
  exec "$BIN" --disable-telemetry
fi
