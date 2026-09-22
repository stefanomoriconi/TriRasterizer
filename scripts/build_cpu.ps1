#Requires -Version 5.1
<#
.SYNOPSIS
    Build the triangle rasterizer with the CPU (OpenMP) backend and run the C tests.

.DESCRIPTION
    Configures a CMake Release build, compiles the shared library + C test
    executable, and runs the CTest suite.  Works on Windows (MSVC), and the
    equivalent Unix script is scripts/build_cpu.sh.

.PARAMETER BuildDir
    CMake binary directory. Defaults to <repo>\build.

.PARAMETER CMake
    Path to the cmake executable. Defaults to 'cmake' on PATH.
#>
param(
    [string]$BuildDir = "$PSScriptRoot\..\build",
    [string]$CMake = "cmake"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "== triangle_rasterizer: CPU backend build ==" -ForegroundColor Cyan
& $CMake -S $Root -B $BuildDir -DTRIANGLE_RASTERIZER_BACKEND=CPU -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }

& $CMake --build $BuildDir --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "CMake build failed ($LASTEXITCODE)" }

# Run the C self-test suite registered by CMake.  'ctest' must run from inside
# the build directory (works with all CMake versions and generators).
Push-Location $BuildDir
try {
    & ctest --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw "C test run failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}

Write-Host "== CPU backend build + tests OK ==" -ForegroundColor Green
