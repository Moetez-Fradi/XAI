#!/usr/bin/env bash
# =============================================================================
# run_unified_paper_bench.sh
#
# Run the attribution suite and the masked-distance research suite on the
# *same* Linux host via HTTP against local release binaries, with N>=5 trials
# per configuration (mean ± std). This closes the two reviewer gaps called out
# in deliverable/main.tex Limitations:
#   1. single-run point estimates
#   2. attribution-on-Apple-Silicon vs masked-on-separate-HTTP-host hardware skew
#
# Usage:
#   ./run_unified_paper_bench.sh                  # start local servers + full paper eval
#   SKIP_SERVERS=1 ./run_unified_paper_bench.sh   # reuse already-running endpoints
#   BENCH_TRIALS=5 BENCH_QUERIES=200 ./run_unified_paper_bench.sh --smoke
#
# Endpoints (defaults match DB_systems README):
#   vanilla Qdrant  -> http://127.0.0.1:6335
#   XQdrant         -> http://127.0.0.1:6333
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
QDRANT_BIN="${QDRANT_BIN:-${REPO_ROOT}/qdrant/target/release/qdrant}"
XQDRANT_BIN="${XQDRANT_BIN:-${REPO_ROOT}/XQdrant/target/release/xqdrant}"

QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6335}"
XQDRANT_URL="${XQDRANT_URL:-http://127.0.0.1:6333}"
BENCH_TRIALS="${BENCH_TRIALS:-5}"
BENCH_QUERIES="${BENCH_QUERIES:-500}"
BENCH_DIMENSIONS="${BENCH_DIMENSIONS:-768}"
SKIP_SERVERS="${SKIP_SERVERS:-0}"
SMOKE=0
STORAGE_ROOT="${STORAGE_ROOT:-${SCRIPT_DIR}/.paper_bench_storage}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"

for arg in "$@"; do
  case "${arg}" in
    --smoke) SMOKE=1 ;;
    --help|-h)
      sed -n '2,25p' "$0"
      exit 0
      ;;
  esac
done

if [[ "${SMOKE}" == "1" ]]; then
  BENCH_QUERIES="${BENCH_QUERIES:-50}"
  BENCH_QUERIES=50
  BENCH_TRIALS=3
  BENCH_DIMENSIONS=768
fi

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }

die() { log "ERROR: $*"; exit 1; }

need_bin() {
  [[ -x "$1" ]] || die "missing executable: $1 (build release binaries first)"
}

wait_http() {
  local url="$1" name="$2" tries="${3:-60}"
  for ((i = 1; i <= tries; i++)); do
    if curl -sf "${url}/readyz" >/dev/null 2>&1 || curl -sf "${url}/" >/dev/null 2>&1; then
      log "${name} ready at ${url}"
      return 0
    fi
    sleep 1
  done
  die "${name} did not become ready at ${url}"
}

start_server() {
  local bin="$1" name="$2" http_port="$3" grpc_port="$4" storage="$5"
  local log_file="${storage}/server.log"
  mkdir -p "${storage}"
  if curl -sf "http://127.0.0.1:${http_port}/" >/dev/null 2>&1; then
    log "${name} already listening on :${http_port}"
    return 0
  fi
  log "Starting ${name} on :${http_port} (storage=${storage})"
  (
    cd "${storage}"
    QDRANT__SERVICE__HTTP_PORT="${http_port}" \
    QDRANT__SERVICE__GRPC_PORT="${grpc_port}" \
    QDRANT__STORAGE__STORAGE_PATH="${storage}/storage" \
    XQDRANT_MASKED_KERNEL="${XQDRANT_MASKED_KERNEL:-gather}" \
      "${bin}" >"${log_file}" 2>&1
  ) &
  echo $! >"${storage}/server.pid"
  wait_http "http://127.0.0.1:${http_port}" "${name}"
}

activate_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    pip install -q -r "${SCRIPT_DIR}/requirements.txt"
  else
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  fi
}

host_line() {
  python3 - <<'PY'
from bench_common import host_fingerprint
h = host_fingerprint()
print(f"{h.get('cpu_model', h.get('processor','?'))} | {h.get('system')} {h.get('machine')} | cores={h.get('cpu_count_logical')}")
PY
}

main() {
  activate_venv
  export BENCH_TRIALS BENCH_QUERIES BENCH_DIMENSIONS QDRANT_URL XQDRANT_URL
  export BENCH_MODE=http
  export PYTHONUNBUFFERED=1
  export XQDRANT_MASKED_KERNEL="${XQDRANT_MASKED_KERNEL:-gather}"

  log "Unified paper bench host: $(host_line)"
  log "trials=${BENCH_TRIALS} queries=${BENCH_QUERIES} dims=${BENCH_DIMENSIONS}"
  log "Qdrant=${QDRANT_URL}  XQdrant=${XQDRANT_URL}"

  if [[ "${SKIP_SERVERS}" != "1" ]]; then
    need_bin "${QDRANT_BIN}"
    need_bin "${XQDRANT_BIN}"
    start_server "${QDRANT_BIN}" "Qdrant" 6335 6336 "${STORAGE_ROOT}/qdrant"
    start_server "${XQDRANT_BIN}" "XQdrant" 6333 6334 "${STORAGE_ROOT}/xqdrant"
  else
    wait_http "${QDRANT_URL}" "Qdrant" 10
    wait_http "${XQDRANT_URL}" "XQdrant" 10
  fi

  local stamp
  stamp="$(date '+%Y-%m-%d_%H-%M-%S')"
  local attr_id="${stamp}__paper_attr_unified_t${BENCH_TRIALS}"
  local tag="paper_unified_t${BENCH_TRIALS}"

  log "=== Attribution suite (HTTP, same host) ==="
  python3 "${SCRIPT_DIR}/bench_suite.py" \
    --mode http \
    --qdrant-url "${QDRANT_URL}" \
    --xqdrant-url "${XQDRANT_URL}" \
    --dataset synthetic \
    --dimensions ${BENCH_DIMENSIONS} \
    --queries "${BENCH_QUERIES}" \
    --trials "${BENCH_TRIALS}" \
    --tests A C D \
    --experiment-id "${attr_id}"

  log "=== Masked suite: Option 1 (SIFT1M if present, else synthetic) ==="
  local opt1_dataset="synthetic"
  if [[ -f "${SCRIPT_DIR}/data/sift/sift_base.fvecs" ]]; then
    opt1_dataset="sift1m"
  fi
  python3 "${SCRIPT_DIR}/research/option1_naive_masked/run_option1_recall_collapse.py" \
    --mode http \
    --qdrant-url "${QDRANT_URL}" \
    --xqdrant-url "${XQDRANT_URL}" \
    --dataset "${opt1_dataset}" \
    --dimensions ${BENCH_DIMENSIONS} \
    --queries "${BENCH_QUERIES}" \
    --trials "${BENCH_TRIALS}"

  log "=== Masked suite: Option 2 hybrid cutoff ==="
  python3 "${SCRIPT_DIR}/research/option2_hybrid_layer_cutoff/run_option2_layer_sweep.py" \
    --mode http \
    --qdrant-url "${QDRANT_URL}" \
    --xqdrant-url "${XQDRANT_URL}" \
    --dataset "${opt1_dataset}" \
    --dimensions ${BENCH_DIMENSIONS} \
    --queries "${BENCH_QUERIES}" \
    --trials "${BENCH_TRIALS}" \
    --experiment-id "${stamp}__option2_${tag}"

  if [[ "${SMOKE}" != "1" ]]; then
    log "=== Masked suite: X2 coherence (offline) ==="
    python3 "${SCRIPT_DIR}/research/xcut2_subspace_coherence/run_xcut2_coherence.py" \
      --dataset "${opt1_dataset}" \
      --dimensions ${BENCH_DIMENSIONS} \
      --queries "${BENCH_QUERIES}" \
      --trials "${BENCH_TRIALS}" || true

    log "=== Masked suite: Option 4 alpha sweep ==="
    python3 "${SCRIPT_DIR}/research/option4_weighted_blend/run_option4_alpha_sweep.py" \
      --mode http \
      --qdrant-url "${QDRANT_URL}" \
      --xqdrant-url "${XQDRANT_URL}" \
      --dataset "${opt1_dataset}" \
      --dimensions ${BENCH_DIMENSIONS} \
      --queries "${BENCH_QUERIES}" \
      --trials "${BENCH_TRIALS}" || true
  fi

  log "Done. Copy new plots from experiments/*${tag}*/plots and experiments/${attr_id}/plots"
  log "into deliverable/figures/, then refresh numeric claims in deliverable/main.tex."
}

main "$@"
