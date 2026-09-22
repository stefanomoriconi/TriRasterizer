#Requires -Version 5.1
<#
.SYNOPSIS
    Build the triangle rasterizer with the CUDA (GPU) backend.

.DESCRIPTION
    Requires the NVIDIA CUDA toolkit and a CMake that supports the CUDA
    language.  Configure a Release build with the CUDA backend, compile the
    shared library + C tests, and run the CTest suite.  On machines without
    a CUDA toolkit the CPU script (scripts/build_cpu.ps1) is the supported
    path; CMake will emit a clear FATAL_ERROR if you request CUDA here.

.PARAMETER BuildDir
    CMake binary directory. Defaults to <repo>\build-cuda.

.PARAMETER CudaArchs
    Comma-separated CMAKE_CUDA_ARCHITECTURES (default: "61;70;75;80;86;89;90").

.PARAMETER CMake
    Path to the cmake executable. Defaults to 'cmake' on PATH.
#>
param(
    [string]$BuildDir = "$PSScriptRoot\..\build-cuda",
    [string]$CudaArchs = "61;70;75;80;86;89;90",
    [string]$CMake = "cmake"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "== triangle_rasterizer: CUDA backend build ==" -ForegroundColor Cyan
& $CMake -S $Root -B $BuildDir `
    -DTRIANGLE_RASTERIZER_BACKEND=CUDA `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CUDA_ARCHITECTURES=$CudaArchs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }

& $CMake --build $BuildDir --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "CMake build failed ($LASTEXITCODE)" }

Push-Location $BuildDir
try {
    & ctest --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw "C test run failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}

Write-Host "== CUDA backend build + tests OK ==" -ForegroundColor Green
