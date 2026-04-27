#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# ── Clean build ───────────────────────────────────────────────────────────────
echo "[run.sh] Cleaning build directory..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

echo "[run.sh] Configuring with CMake..."
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release

echo "[run.sh] Building..."
cmake --build "${BUILD_DIR}" -- -j"$(nproc)"

# ── Run controller ────────────────────────────────────────────────────────────
echo "[run.sh] Running MPCC controller..."
cd "${SCRIPT_DIR}"
"${BUILD_DIR}/MPCC_controller_cpp"

# ── Playback / visualisation ──────────────────────────────────────────────────
RESULTS_DIR="${SCRIPT_DIR}/results"
LATEST_RUN_DIR=""
if [[ -d "${RESULTS_DIR}" ]]; then
    LATEST_RUN_DIR="$(find "${RESULTS_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | awk 'NR==1{print $2}')"
fi

PLAYBACK_ARGS=(--track-folder "${SCRIPT_DIR}/raceline")
if [[ -n "${LATEST_RUN_DIR}" && -f "${LATEST_RUN_DIR}/states_ctrls.csv" ]]; then
    PLAYBACK_ARGS+=(--results-dir "${LATEST_RUN_DIR}")
fi

if [[ "${1:-}" == "gif" ]]; then
    mkdir -p "${SCRIPT_DIR}/plots"
    PLAYBACK_ARGS+=(--gif "${SCRIPT_DIR}/plots/playback.gif" --no-show)
fi

echo "[run.sh] Running playback script..."
python3 "${SCRIPT_DIR}/scripts/playback_dashboard_csv.py" "${PLAYBACK_ARGS[@]}"
