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
GIF_ARGS=""
if [[ "${1:-}" == "gif" ]]; then
    mkdir -p "${SCRIPT_DIR}/plots"
    GIF_ARGS="--gif ${SCRIPT_DIR}/plots/playback.gif --no-show"
fi

echo "[run.sh] Running playback script..."
python3 "${SCRIPT_DIR}/scripts/playback_dashboard_csv.py" ${GIF_ARGS}
