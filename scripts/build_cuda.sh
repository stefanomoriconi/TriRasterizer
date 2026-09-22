#!/usr/bin/env bash
# Build the triangle rasterizer with the CUDA (GPU) backend.
# Requires the NVIDIA CUDA toolkit and a CMake with CUDA language support.
#
# Usage:
#   scripts/build_cuda.sh [build_dir] [cuda_archs]
#
#   build_dir   defaults to <repo>/build-cuda
#   cuda_archs  defaults to "61;70;75;80;86;89;90"
#
# Environment:
#   CMAKE - optional path to the cmake executable (defaults to 'cmake' on PATH)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
BUILD_DIR="${1:-$ROOT/build-cuda}"
CUDA_ARCHS="${2:-61;70;75;80;86;89;90}"
CMAKE="${CMAKE:-cmake}"

echo "== triangle_rasterizer: CUDA backend build =="
"$CMAKE" -S "$ROOT" -B "$BUILD_DIR" \
    -DTRIANGLE_RASTERIZER_BACKEND=CUDA \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS"
"$CMAKE" --build "$BUILD_DIR" --config Release --parallel

(cd "$BUILD_DIR" && ctest --output-on-failure)

echo "== CUDA backend build + tests OK =="
