#!/usr/bin/env bash
# =============================================================================
# run_bench.sh — System-isolated orchestration for the XQdrant benchmark suite
#
# Handles:
#   • OS page-cache drop (with safe fallback)
#   • CPU affinity pinning via taskset (or numactl when available)
#   • Python virtual-environment bootstrap
#   • Invocation of bench_suite.py with reproducible environment variables
#
# Usage:
#   ./run_bench.sh                          # simulated mode (no server)
#   ./run_bench.sh --mode http              # live Qdrant/XQdrant endpoints
#   BENCH_CPU_CORES="2,3" ./run_bench.sh    # pin to specific cores
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# Configurable parameters (override via environment)
# ---------------------------------------------------------------------------
BENCH_MODE="${BENCH_MODE:-simulated}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
XQDRANT_URL="${XQDRANT_URL:-${QDRANT_URL}}"
BENCH_DIMENSIONS="${BENCH_DIMENSIONS:-768 1536}"
BENCH_QUERIES="${BENCH_QUERIES:-500}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
BENCH_CPU_CORES="${BENCH_CPU_CORES:-}"   # e.g. "0,1,2,3" or "0-3"
VALIDATE_HTTP="${VALIDATE_HTTP:-0}"      # set to 1 to run endpoint probes before benchmark

# Forward any extra CLI args to bench_suite.py (e.g. --tests A B)
# Note: macOS bash 3.2 + `set -u` treats "${array[@]}" on an empty array as
# unbound, so we only append when args are actually present (see main()).

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    # stderr so command substitutions (e.g. exec_prefix=...) stay clean
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

detect_cpu_cores() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        sysctl -n hw.logicalcpu 2>/dev/null || echo 4
    else
        nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4
    fi
}

detect_physical_cores() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        sysctl -n hw.physicalcpu 2>/dev/null || detect_cpu_cores
    else
        nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4
    fi
}

auto_select_cores() {
    local total
    total="$(detect_physical_cores)"
    [[ "${total}" -lt 1 ]] && total=1
    local cores=""
    for ((i = 0; i < total; i++)); do
        [[ -n "${cores}" ]] && cores+=","
        cores+="${i}"
    done
    echo "${cores}"
}

drop_page_caches() {
    log "Attempting to drop OS page caches for cold-start reproducibility..."
    if [[ "$(uname -s)" == "Linux" ]] && [[ -w /proc/sys/vm/drop_caches ]]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches
        log "Page caches dropped successfully."
    elif [[ "$(uname -s)" == "Linux" ]]; then
        log "WARNING: Cannot write /proc/sys/vm/drop_caches (need root)."
        log "         Run manually before benchmarking:"
        log "           sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
    else
        log "NOTE: Page-cache drop via /proc/sys/vm/drop_caches is Linux-only."
        log "      On macOS, restart the Qdrant process or reboot for cold-cache runs."
    fi
}

disable_turbo_hint() {
    # Best-effort reminder; actual MSR writes require root and are platform-specific.
    if [[ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]] 2>/dev/null; then
        local prev
        prev="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
        echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
        log "Intel Turbo Boost disabled for this session (was: ${prev})."
    else
        log "NOTE: To reduce Turbo Boost variance, run as root:"
        log "        echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo"
    fi
}

build_exec_prefix() {
    local cores="${BENCH_CPU_CORES}"
    if [[ -z "${cores}" ]]; then
        cores="$(auto_select_cores)"
    fi
    log "Pinning benchmark to CPU cores: ${cores}"

    if command -v taskset &>/dev/null; then
        printf 'taskset -c %s' "${cores}"
    elif command -v gtaskset &>/dev/null; then
        # Homebrew util-linux on macOS installs g-prefixed binaries
        printf 'gtaskset -c %s' "${cores}"
    elif command -v numactl &>/dev/null; then
        printf 'numactl --physcpubind=%s' "${cores}"
    else
        if [[ "$(uname -s)" == "Darwin" ]]; then
            log "NOTE: CPU pinning unavailable on macOS (no taskset/numactl)."
            log "      Optional: brew install util-linux  # provides gtaskset"
        else
            log "WARNING: Neither taskset nor numactl found; running without CPU pinning."
        fi
        printf ''
    fi
}

setup_venv() {
    # Prefer uv when available: uv-created venvs intentionally ship without pip,
    # so calling `pip` would fall through to the system interpreter (which Arch
    # blocks via PEP 668 "externally-managed-environment"). Use `uv pip` instead.
    if command -v uv &>/dev/null; then
        if [[ ! -d "${VENV_DIR}" ]]; then
            log "Creating Python virtual environment at ${VENV_DIR} (uv)"
            uv venv "${VENV_DIR}"
        fi
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
        log "Installing dependencies via uv pip"
        uv pip install --quiet -r "${SCRIPT_DIR}/requirements.txt"
        return
    fi

    if [[ ! -d "${VENV_DIR}" ]]; then
        log "Creating Python virtual environment at ${VENV_DIR}"
        python3 -m venv "${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    # Use `python3 -m pip` so we always target the venv interpreter, never a
    # stray system `pip` on PATH.
    python3 -m pip install --quiet --upgrade pip
    python3 -m pip install --quiet -r "${SCRIPT_DIR}/requirements.txt"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    # Extra bench_suite.py args passed through from the shell invocation.
    local extra_args=("$@")

    log "XQdrant Benchmark Orchestrator"
    log "Working directory: ${SCRIPT_DIR}"

    setup_venv
    drop_page_caches
    disable_turbo_hint

    mkdir -p "${SCRIPT_DIR}/results" "${SCRIPT_DIR}/plots" "${SCRIPT_DIR}/.mplconfig" "${SCRIPT_DIR}/experiments"

    # Keep matplotlib cache inside the project (avoids permission issues on macOS)
    export MPLCONFIGDIR="${SCRIPT_DIR}/.mplconfig"

    local exec_prefix
    exec_prefix="$(build_exec_prefix)"

    export BENCH_MODE QDRANT_URL XQDRANT_URL BENCH_DIMENSIONS BENCH_QUERIES

    local -a bench_cmd=(
        python3 "${SCRIPT_DIR}/bench_suite.py"
        --mode "${BENCH_MODE}"
        --qdrant-url "${QDRANT_URL}"
        --xqdrant-url "${XQDRANT_URL}"
        --dimensions ${BENCH_DIMENSIONS}
        --queries "${BENCH_QUERIES}"
    )
    if ((${#extra_args[@]} > 0)); then
        bench_cmd+=("${extra_args[@]}")
    fi

    if [[ "${BENCH_MODE}" == "http" && "${VALIDATE_HTTP}" == "1" ]]; then
        log "Running HTTP pre-flight validation..."
        python3 "${SCRIPT_DIR}/validate_http.py" \
            --qdrant-url "${QDRANT_URL}" \
            --xqdrant-url "${XQDRANT_URL}"
    fi

    local -a prefix=()
    if [[ -n "${exec_prefix}" ]]; then
        # shellcheck disable=SC2206
        prefix=(${exec_prefix})
    fi

    if ((${#prefix[@]} > 0)); then
        log "Executing: ${prefix[*]} ${bench_cmd[*]}"
        "${prefix[@]}" "${bench_cmd[@]}"
    else
        log "Executing: ${bench_cmd[*]}"
        "${bench_cmd[@]}"
    fi

    log "Done. Experiments: ${SCRIPT_DIR}/experiments/"
}

main "$@"
