#!/usr/bin/env bash
# Build the triangle rasterizer with the CPU (OpenMP) backend and run the C tests.
# Works on Linux and macOS (glibc or libc++ toolchains, clang or gcc).
#
# Usage:
#   scripts/build_cpu.sh [build_dir]
#
# Environment:
#   CMAKE   - optional path to the cmake executable (defaults to 'cmake' on PATH)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
BUILD_DIR="${1:-$ROOT/build}"
CMAKE="${CMAKE:-cmake}"

echo "== triangle_rasterizer: CPU backend build =="
"$CMAKE" -S "$ROOT" -B "$BUILD_DIR" \
    -DTRIANGLE_RASTERIZER_BACKEND=CPU \
    -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build "$BUILD_DIR" --config Release --parallel

# Run the C self-test suite registered by CMake (run from the build dir).
(cd "$BUILD_DIR" && ctest --output-on-failure)

echo "== CPU backend build + tests OK =="
