#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — build, simulate both scenarios, generate playback GIFs
#
# Usage:
#   ./run.sh          # build + run (GIFs are created inside each result folder)
#   ./run.sh rebuild  # force clean rebuild before running
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Make local HSL shared libraries visible for IPOPT deferred loading.
HSL_LIB_DIR="${HOME}/ThirdParty-HSL/.libs"
if [[ -d "${HSL_LIB_DIR}" ]]; then
    export LD_LIBRARY_PATH="${HSL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    echo "[run.sh] HSL libs: ${HSL_LIB_DIR}"
fi

# ── Build ─────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "rebuild" ]] || [[ ! -f "${BUILD_DIR}/MPCC_controller_cpp" ]]; then
    echo "[run.sh] Configuring…"
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    echo "[run.sh] Building…"
    cmake --build "${BUILD_DIR}" -- -j"$(nproc)"
else
    echo "[run.sh] Incremental build…"
    cmake --build "${BUILD_DIR}" -- -j"$(nproc)"
fi

# ── Run ───────────────────────────────────────────────────────────────────────
# main.cpp runs both scenarios internally and generates plots + GIFs
echo "[run.sh] Running MPCC (both scenarios)…"
cd "${SCRIPT_DIR}"
"${BUILD_DIR}/MPCC_controller_cpp"

echo "[run.sh] Done."
