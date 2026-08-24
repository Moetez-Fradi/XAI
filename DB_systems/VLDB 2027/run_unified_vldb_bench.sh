#!/usr/bin/env bash
# =============================================================================
# run_unified_vldb_bench.sh
#
# VLDB 2027 unified harness: start Qdrant + XQdrant, run every paper panel,
# write timestamped CSVs under experiments/.
#
# Usage:
#   ./run_unified_vldb_bench.sh --smoke --simulated   # no servers
#   ./run_unified_vldb_bench.sh --smoke               # tiny HTTP
#   ./run_unified_vldb_bench.sh --expansion           # N=20k Q=40 T=5 (paper §5.1)
#   ./run_unified_vldb_bench.sh --full                # SIFT1M / MiniLM 100k (long)
#   SKIP_SERVERS=1 ./run_unified_vldb_bench.sh --expansion
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
QDRANT_BIN="${QDRANT_BIN:-${REPO_ROOT}/qdrant/target/release/qdrant}"
XQDRANT_BIN="${XQDRANT_BIN:-${REPO_ROOT}/XQdrant/target/release/xqdrant}"

QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6335}"
XQDRANT_URL="${XQDRANT_URL:-http://127.0.0.1:6333}"
SKIP_SERVERS="${SKIP_SERVERS:-0}"
STORAGE_ROOT="${STORAGE_ROOT:-${SCRIPT_DIR}/.paper_bench_storage}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
MODE="http"
PROFILE="expansion"

for arg in "$@"; do
  case "${arg}" in
    --smoke) PROFILE="smoke" ;;
    --expansion) PROFILE="expansion" ;;
    --full) PROFILE="full" ;;
    --simulated) MODE="simulated" ;;
    --help|-h)
      sed -n '2,20p' "$0"
      exit 0
      ;;
  esac
done

case "${PROFILE}" in
  smoke)
    N=512; Q=8; T=1; DIMS=768
    ;;
  expansion)
    N=20000; Q=40; T=5; DIMS=768
    ;;
  full)
    N=""; Q=500; T=5; DIMS=768
    ;;
esac

export BENCH_TRIALS="${T}"
export BENCH_QUERIES="${Q}"
export BENCH_DIMENSIONS="${DIMS}"
export BENCH_MODE="${MODE}"
export PYTHONUNBUFFERED=1
export XQDRANT_MASKED_KERNEL="${XQDRANT_MASKED_KERNEL:-gather}"
export MPLCONFIGDIR="${SCRIPT_DIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/experiments"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H-%M-%S')" "$*" >&2; }
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
    python3 -m pip install -q --upgrade pip
    python3 -m pip install -q -r "${SCRIPT_DIR}/requirements.txt"
  else
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  fi
}

py() { python3 "$@"; }

n_flag() {
  if [[ -n "${N}" ]]; then
    echo --num-vectors "${N}"
  fi
}

main() {
  activate_venv
  log "profile=${PROFILE} mode=${MODE} N=${N:-full} Q=${Q} T=${T}"

  if [[ "${MODE}" == "http" && "${SKIP_SERVERS}" != "1" ]]; then
    need_bin "${QDRANT_BIN}"
    need_bin "${XQDRANT_BIN}"
    start_server "${QDRANT_BIN}" "Qdrant" 6335 6336 "${STORAGE_ROOT}/qdrant"
    start_server "${XQDRANT_BIN}" "XQdrant" 6333 6334 "${STORAGE_ROOT}/xqdrant"
  elif [[ "${MODE}" == "http" ]]; then
    wait_http "${QDRANT_URL}" "Qdrant" 10
    wait_http "${XQDRANT_URL}" "XQdrant" 10
  fi

  local sift_ds="synthetic"
  if [[ -f "${SCRIPT_DIR}/data/sift/sift_base.fvecs" ]]; then
    sift_ds="sift1m"
  fi

  log "=== Fig 2 cost model ==="
  py research/cost_model/plot_cost_model.py || true

  log "=== Figs 3–5 attribution suite ==="
  py bench_suite.py \
    --mode "${MODE}" \
    --qdrant-url "${QDRANT_URL}" --xqdrant-url "${XQDRANT_URL}" \
    --dataset synthetic --dimensions "${DIMS}" \
    --num-vectors "${N:-10000}" --queries "${Q}" --trials "${T}" \
    --tests A C D || true

  log "=== Fig 6 systems load ==="
  py research/systems_load/run_systems_load.py \
    --mode "${MODE}" --qdrant-url "${QDRANT_URL}" --xqdrant-url "${XQDRANT_URL}" \
    --dataset synthetic --dimensions "${DIMS}" \
    $(n_flag) --queries "${Q}" --trials "${T}" || true

  log "=== Fig 7 M1 ==="
  py research/option1_naive_masked/run_option1_recall_collapse.py \
    --mode "${MODE}" --qdrant-url "${XQDRANT_URL}" --xqdrant-url "${XQDRANT_URL}" \
    --dataset "${sift_ds}" --dimensions "${DIMS}" \
    $(n_flag) --queries "${Q}" --trials "${T}" || true

  log "=== Fig 8 MiniLM/GIST M1 ==="
  local corpora=()
  if [[ -f "${SCRIPT_DIR}/data/minilm/embeddings.npy" ]]; then corpora+=(minilm); fi
  if [[ -f "${SCRIPT_DIR}/data/gist/gist_base.fvecs" ]]; then corpora+=(gist1m); fi
  if [[ ${#corpora[@]} -eq 0 ]]; then corpora=(synthetic); fi
  py research/cross_corpus/run_minilm_gist_m1.py \
    --mode "${MODE}" --qdrant-url "${XQDRANT_URL}" --xqdrant-url "${XQDRANT_URL}" \
    --corpora "${corpora[@]}" --dataset "${corpora[0]}" \
    $(n_flag) --queries "${Q}" --trials "${T}" || true

  log "=== Figs 9–11 M2 ==="
  py research/option2_hybrid_layer_cutoff/run_option2_layer_sweep.py \
    --mode "${MODE}" --qdrant-url "${XQDRANT_URL}" --xqdrant-url "${XQDRANT_URL}" \
    --dataset "${sift_ds}" --dimensions "${DIMS}" \
    $(n_flag) --queries "${Q}" --trials "${T}" \
    --ratios 0.25 0.5 0.75 --layers 0 1 2 || true

  log "=== Fig 12 scale×ef ==="
  local ns_args=(--ns 512)
  if [[ "${PROFILE}" == "expansion" ]]; then ns_args=(--ns 10000 20000); fi
  if [[ "${PROFILE}" == "full" ]]; then ns_args=(--ns 10000 50000 100000); fi
  if [[ "${PROFILE}" == "smoke" ]]; then ns_args=(--ns 512); fi
  py research/scale_ef/run_scale_ef.py \
    --mode "${MODE}" --qdrant-url "${XQDRANT_URL}" --xqdrant-url "${XQDRANT_URL}" \
    --dataset "${sift_ds}" "${ns_args[@]}" --efs 64 128 \
    --queries "${Q}" --trials "${T}" || true

  log "=== Fig 13 focus baselines ==="
  py research/focus_baselines/run_focus_baselines.py \
    --mode "${MODE}" --qdrant-url "${XQDRANT_URL}" --xqdrant-url "${XQDRANT_URL}" \
    --dataset "${sift_ds}" --dimensions "${DIMS}" \
    $(n_flag) --queries "${Q}" --trials "${T}" || true

  log "=== Fig 14 K1 (example CSV if no Rust bench) ==="
  py research/xcut1_gather_vs_repack/plot_xcut1_kernel_bench.py --make-example || true

  log "=== Figs 15–16 C1 ==="
  py research/xcut2_subspace_coherence/run_xcut2_coherence.py \
    --dataset "${sift_ds}" --dimensions "${DIMS}" \
    $(n_flag) --queries "${Q}" --trials "${T}" || true

  log "=== Fig 17 X4 ==="
  py research/xcut4_divergence/run_xcut4_divergence.py \
    --mode "${MODE}" --qdrant-url "${XQDRANT_URL}" --xqdrant-url "${XQDRANT_URL}" \
    --dataset "${sift_ds}" --dimensions "${DIMS}" \
    $(n_flag) --queries "${Q}" || true

  log "=== Fig 18 V1 ==="
  py research/xcut3_verify_pass/run_xcut3_verify.py \
    --mode "${MODE}" --qdrant-url "${XQDRANT_URL}" --xqdrant-url "${XQDRANT_URL}" \
    --dataset "${sift_ds}" --dimensions "${DIMS}" \
    $(n_flag) --queries "${Q}" --trials "${T}" || true

  log "=== Fig 19 M3 ==="
  py research/option4_weighted_blend/run_option4_alpha_sweep.py \
    --mode "${MODE}" --qdrant-url "${XQDRANT_URL}" --xqdrant-url "${XQDRANT_URL}" \
    --dataset "${sift_ds}" --dimensions "${DIMS}" \
    $(n_flag) --queries "${Q}" --trials "${T}" || true

  log "=== Fig 20 filter × M2 ==="
  py research/filter_m2/run_filter_m2.py \
    --mode "${MODE}" --qdrant-url "${XQDRANT_URL}" --xqdrant-url "${XQDRANT_URL}" \
    --dataset "${sift_ds}" --dimensions "${DIMS}" \
    $(n_flag) --queries "${Q}" --trials "${T}" || true

  log "=== Fig 21 SQ ==="
  py research/quantization_sq/run_sq_char.py \
    --mode "${MODE}" --qdrant-url "${XQDRANT_URL}" --xqdrant-url "${XQDRANT_URL}" \
    --dataset "${sift_ds}" --dimensions "${DIMS}" \
    $(n_flag) --queries "${Q}" --trials "${T}" || true

  log "=== Fig 22 projected ceiling ==="
  py research/option3_projected_index/run_option3_projected_index.py \
    --mode "${MODE}" --xqdrant-url "${XQDRANT_URL}" \
    --dataset "${sift_ds}" --dimensions "${DIMS}" \
    $(n_flag) --queries "${Q}" || true

  log "=== Fig 23 case study ==="
  py research/case_study/run_marco_case.py \
    --mode "${MODE}" --qdrant-url "${XQDRANT_URL}" --xqdrant-url "${XQDRANT_URL}" \
    $(n_flag) --queries "${Q}" || true

  log "Done. Results under ${SCRIPT_DIR}/experiments/"
  log "Canonical paper figures remain in paper/figures/ (frozen from the submission PDF)."
}

main "$@"
