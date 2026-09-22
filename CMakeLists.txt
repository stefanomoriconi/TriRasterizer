# CMakeLists.txt
# -----------------------------------------------------------------------------
# triangle_rasterizer — a small, data-race-free 2D triangle rasteriser.
#
# Multi-platform, multi-backend shared library.
#
#   Backend selection (option TRIANGLE_RASTERIZER_BACKEND):
#     AUTO  - use CUDA if a CUDA toolchain is found, otherwise CPU          (default)
#     CPU   - always build the OpenMP/CPU backend
#     CUDA  - build the NVIDIA CUDA backend (requires CUDA toolkit + CMake>=3.8)
#
#   The exported C ABI is IDENTICAL for both backends; only the rasterisation
#   implementation (one source file) differs. The Python ctypes wrapper works
#   against either without modification.
#
# Typical usage (CPU, any OS):
#   cmake -S . -B build
#   cmake --build build --config Release
#
# Typical usage (CUDA, any OS with an NVIDIA toolkit):
#   cmake -S . -B build -DTRIANGLE_RASTERIZER_BACKEND=CUDA
#   cmake --build build --config Release
# -----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.15)

project(triangle_rasterizer
        VERSION 1.3.0
        DESCRIPTION "Data-race-free, parallelisable 2D triangle rasteriser (CPU + CUDA backends)."
        LANGUAGES C)

# ---- Options ------------------------------------------------------------------
set(TRIANGLE_RASTERIZER_BACKEND "AUTO"
    CACHE STRING "Rasteriser backend: AUTO, CPU, or CUDA")
set_property(CACHE TRIANGLE_RASTERIZER_BACKEND PROPERTY STRINGS AUTO CPU CUDA)

option(BUILD_SHARED_LIBS           "Build a shared library (recommended)" ON)
option(TRIANGLE_RASTERIZER_TESTS   "Build and register the C self-tests"  ON)
option(TRIANGLE_RASTERIZER_OPENMP  "Enable OpenMP parallelism (CPU backend)" ON)

# ---- Language standard --------------------------------------------------------
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Default an install prefix on single-config generators so `install` is sane.
if(NOT CMAKE_INSTALL_PREFIX_INIT)
  set(CMAKE_INSTALL_PREFIX_INIT "${CMAKE_BINARY_DIR}/install")
endif()

# ---- Resolve backend ----------------------------------------------------------
set(TRI_BACKEND_ACTUAL "CPU")

if(TRIANGLE_RASTERIZER_BACKEND STREQUAL "CUDA" OR
   TRIANGLE_RASTERIZER_BACKEND STREQUAL "AUTO")
    # Probe for a CUDA compiler without committing to it.
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        # Broad SM coverage by default; override with -DCMAKE_CUDA_ARCHITECTURES=...
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "61;70;75;80;86;89;90")
        endif()
        set(TRI_BACKEND_ACTUAL "CUDA")
    elseif(TRIANGLE_RASTERIZER_BACKEND STREQUAL "CUDA")
        message(FATAL_ERROR
            "TRIANGLE_RASTERIZER_BACKEND=CUDA was requested but no CUDA toolchain "
            "was found. Install the NVIDIA CUDA toolkit or set CMAKE_CUDA_COMPILER. "
            "(To build the CPU backend instead, pass -DTRIANGLE_RASTERIZER_BACKEND=CPU)")
    endif()
endif()

message(STATUS "triangle_rasterizer: selected backend = ${TRI_BACKEND_ACTUAL}")

# ---- Library target -----------------------------------------------------------
if(TRI_BACKEND_ACTUAL STREQUAL "CUDA")
    add_library(triangle_rasterizer
        src/raster_common.c
        src/backend_cuda.cu)
    set_target_properties(triangle_rasterizer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
else()
    add_library(triangle_rasterizer
        src/raster_common.c
        src/backend_cpu.c)
endif()

target_include_directories(triangle_rasterizer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Compile flags (multi-platform).
if(MSVC)
    target_compile_definitions(triangle_rasterizer PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(triangle_rasterizer PRIVATE /W4 /permissive-)
    if(BUILD_SHARED_LIBS)
        target_compile_definitions(triangle_rasterizer PUBLIC TRIANGLE_RASTERIZER_BUILDING_DLL)
    endif()
else()
    target_compile_options(triangle_rasterizer PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
    target_compile_definitions(triangle_rasterizer PRIVATE _POSIX_C_SOURCE=200809L)
endif()

target_compile_definitions(triangle_rasterizer PRIVATE TRIANGLE_RASTERIZER_BUILDING)

# CPU backend: link OpenMP when requested and available.
if(TRI_BACKEND_ACTUAL STREQUAL "CPU" AND TRIANGLE_RASTERIZER_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_C_FOUND)
        target_link_libraries(triangle_rasterizer PRIVATE OpenMP::OpenMP_C)
        target_compile_definitions(triangle_rasterizer PRIVATE TRIANGLE_RASTERIZER_WITH_OPENMP)
    endif()
endif()

# CUDA backend: link the CUDA runtime (via the CMAKE_CUDA_RUNTIME_LIBRARY
# property so it is found on all of Windows/macOS/Linux).
if(TRI_BACKEND_ACTUAL STREQUAL "CUDA")
    # Prefer a statically-linked CUDA runtime where possible so the .so/.dylib
    # / .dll does not require the CUDA runtime DLL to be on the PATH at runtime.
    # CMake links the CUDA runtime automatically based on this property;
    # STATIC is preferred so the shared library does not require the CUDA
    # runtime shared object on the runtime PATH.
    set_property(TARGET triangle_rasterizer PROPERTY CUDA_RUNTIME_LIBRARY STATIC)
endif()

# Shared library hygiene (symbol visibility is handled by the header macros).
if(BUILD_SHARED_LIBS)
    set_target_properties(triangle_rasterizer PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()

# ---- Install ------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS triangle_rasterizer
    EXPORT triangle_rasterizerTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/triangle_rasterizer.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT triangle_rasterizerTargets
    FILE triangle_rasterizerConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/triangle_rasterizer)

# ---- Tests --------------------------------------------------------------------
if(TRIANGLE_RASTERIZER_TESTS)
    enable_testing()
    add_executable(triangle_rasterizer_test tests/test_rasterizer.c)
    target_link_libraries(triangle_rasterizer_test PRIVATE triangle_rasterizer)
    add_test(NAME triangle_rasterizer_test COMMAND triangle_rasterizer_test)
    # Report the active backend at test time for diagnostics.
    set_tests_properties(triangle_rasterizer_test PROPERTIES ENVIRONMENT "TRI_PRINT_BACKEND=1")
endif()

# ---- Package summary ----------------------------------------------------------
message(STATUS "---------------------------------------------------------------")
message(STATUS "triangle_rasterizer ${PROJECT_VERSION}")
message(STATUS "  backend          : ${TRI_BACKEND_ACTUAL}")
message(STATUS "  shared library   : ${BUILD_SHARED_LIBS}")
message(STATUS "  OpenMP           : ${TRIANGLE_RASTERIZER_OPENMP}")
message(STATUS "  tests            : ${TRIANGLE_RASTERIZER_TESTS}")
message(STATUS "---------------------------------------------------------------")
