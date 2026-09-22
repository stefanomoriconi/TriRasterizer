# Triangle Rasterizer

A fast, **data-race-free** 2D triangle rasterizer written in **C11** with an
optional **CUDA** backend.  It is designed to be:

* **Thread-safe and deterministic** — every scanline / pixel is independent, so
  results are bit-identical regardless of the thread count *and* between the
  CPU and CUDA backends.
* **Multi-platform** — builds on **Windows, macOS, and Linux** with a
  plain C toolchain (CPU) or the NVIDIA CUDA toolkit (GPU).
* **Stable C ABI** — a small, additive-only C interface that is safe to expose
  to Python, other languages, or a C++ host application.
* **Easy to embed** — one header, one shared library, a NumPy-friendly Python
  package on top, ready-to-run examples, and a full test suite (C + Python).

> Status: **v1.3.0** (optional fixed-point precision added; v1.1.0 / v1.2.0 APIs fully preserved).  Phase 1 (CPU) is fully implemented,
> tested, and packaged.  The CUDA backend is implemented and selected at build time; it is exercised in CI where a GPU toolchain is available.

---

## Table of Contents

1. [Features](#features)
2. [How it works (performance model)](#how-it-works-performance-model)
3. [Why it is free of data races](#why-it-is-free-of-data-races)
4. [Backends: CPU (OpenMP) and CUDA](#backends-cpu-openmp-and-cuda)
5. [Project layout](#project-layout)
6. [Building](#building)
   - [Windows](#windows)
   - [Linux / macOS](#linux--macos)
   - [CUDA backend](#cuda-backend)
   - [CMake options](#cmake-options)
7. [Installing the Python package](#installing-the-python-package)
8. [Using it from Python](#using-it-from-python)
9. [Flags at a glance](#flags-at-a-glance)
10. [Anti-aliasing (v1.2.0)](#anti-aliasing-v120)
11. [Precision (v1.3.0)](#precision-v130)
12. [Using it from C](#using-it-from-c)
13. [Error codes](#error-codes)
14. [Running the tests](#running-the-tests)
15. [Examples](#examples)
16. [Continuous integration](#continuous-integration)
17. [Design notes & limitations](#design-notes--limitations)
18. [License](#license)

---

## Features

* Software rasterization of filled 2D triangles (flat colour + per-triangle
  alpha).
* **Two interchangeable backends** behind a single C ABI:
  - **CPU** — OpenMP scanline parallelism (portable, no GPU required).
  - **CUDA** — one thread per pixel (opt-in, requires the NVIDIA toolkit).
  - Automatic **fallback to CPU** when CUDA is requested but unavailable.
* Deterministic output: identical pixels for any thread count and identical
  pixels across the CPU and CUDA backends.
* **Optional anti-aliasing** (v1.2.0): exact analytical coverage or 2×2 /
  4-tap rotated supersampling, selectable per call; off by default so the
  legacy hard-edge path is untouched.
* **Optional fixed-point precision** (v1.3.0): an opt-in Q24.8
  fixed-point scanline path (`PRECISION_FIXED8`) alongside the float path, for
  bit-exact deterministic results; float stays the default.
* Stable C ABI with a versioned header and explicit error codes.
* `uint8` **RGB / RGBA** input and output via NumPy arrays in Python.
* Cross-platform builds (Windows / Linux / macOS) and a Python package
  installable with `pip`.

## How it works (performance model)

The core loop is a **scanline / top-down triangle traversal** (the classic
"top-left fill rule"):

1. Sort the triangle vertices by `y` into `t`, `m`, `b`.
2. Split into a "bottom" and an (optional) "right" sub-triangle.
3. For every scanline inside the triangle, compute the two edge `x`
   intersections with cheap incremental differences (no division per pixel).
4. Fill the horizontal span `[x_left, x_right)` with the triangle colour,
   applying per-triangle alpha as `out = dst * (1 - a) + src * a`
   (a fast path is used when `a >= 1.0`).

The loop is parallelised over scanlines (CPU) or pixels (CUDA).  Because each
pixel is written by exactly one thread, and each pixel's computation depends
only on the triangle parameters (not on neighbouring pixels), there is **no
shared mutable state** and therefore **no data race**.

> **Colour convention:** RGB components are `0..255`; alpha is `0..1`.  Only
> the first three channels (RGB) are written; the 4th channel (alpha) of an
> RGBA buffer is left untouched.  For a 4-channel image with the same
> semantics, blend the RGB and then write your own alpha if needed.

### CPU micro-optimisations

The CPU (OpenMP) backend applies a few **bit-identical** micro-optimisations
on the hot path (all guarded, and every one is provably the same output as the
reference math):

* **Per-row span hoisting (AA modes).** For `AA_ANALYTICAL` and the SSAA
  modes the per-sample x-spans are computed **once per scanline** and reused
  for every pixel on that row, instead of re-deriving them per pixel. Each
  pixel then only does a few compares. This is the single largest CPU speed-up
  for the anti-aliased paths.
* **SSE2 opaque constant fill.** When a span is a pure opaque fill
  (`AA_NONE` with `a >= 1.0`), the constant RGB triple is written in wide SSE2
  lanes (4 RGBA / 2 RGB pixels at a time) instead of byte-by-byte, while still
  leaving the 4th RGBA alpha byte untouched. The semi-transparent blend path
  stays on the scalar per-pixel writer because it is genuinely per-channel and
  must remain byte-exact.

None of these change the result: `AA_NONE` output is byte-identical to the
legacy path, and the AA paths use exactly the same span/coverage expressions as
before, merely hoisted out of the pixel loop.

## Why it is free of data races

* **CPU (OpenMP):** the parallel region is a `for` over scanlines with a
  `default(none)` clause, so every variable is either captured explicitly or
  is a loop-local.  Each scanline writes to a disjoint `y`-row of the image.
  Two triangles can overlap, but they are rasterised sequentially (painter's
  order, input order), so there is no write from two threads to the same
  pixel at the same time.
* **CUDA:** the kernel assigns **one pixel per thread**.  A pixel is visited
  by at most one thread (grid-stride loop over the bounding box), so two
  threads never write the same pixel.  No shared memory, no atomics.
* **Host side:** the triangle batch is built once on the host (a single,
  locked, copy of the input list) and is *read-only* for the lifetime of the
  call.  No thread ever mutates the batch.

The upshot: **no locks, no atomics, no critical sections on the hot path**,
and the output is **bit-identical** across thread counts *and* backends.

## Backends: CPU (OpenMP) and CUDA

The library is a single shared library; the backend is chosen at **build
time** with a CMake option (`AUTO`, `CPU`, or `CUDA`) — see
[CMake options](#cmake-options).

| Backend | Selected when | Requirements | Notes |
|---------|---------------|--------------|-------|
| CPU (OpenMP) | `AUTO` or `CPU`, always as fallback | C11 + OpenMP (MSVC / GCC / Clang) | Portable; default. |
| CUDA | `AUTO` (if a toolkit is found) or `CUDA` | NVIDIA CUDA toolkit + compatible GPU | One thread per pixel. |

* **`AUTO`** (default): the CMake configure step probes for a CUDA toolkit.
  If one is found, the CUDA backend is compiled in; otherwise the CPU backend
  is used and the build proceeds normally.
* **Runtime fallback:** even when the CUDA backend is compiled in, if no CUDA
  device is present at load time the rasterizer transparently falls back to
  the CPU scanline code, so the same library works on GPU and non-GPU hosts.
* **Query the active backend:** `triangle_rasterizer_backend()` (C) and
  `triangle_rasterizer.backend()` (Python) return a short string such as
  `"cpu (OpenMP)"`, `"cpu"`, or `"cuda"`.
* **Determinism:** the CUDA kernel and the CPU scanline code use the same
  edge equations and the same fill order, so their output is **bit-identical**
  for the same input.

### CUDA kernel (overview)

Each thread owns one pixel inside the triangle's bounding box and tests
whether the pixel lies inside the triangle using the standard barycentric /
edge-sign test.  Because each pixel is owned by exactly one thread, the kernel
needs no shared memory, no atomics, and no inter-thread synchronisation —
which is exactly what keeps it data-race-free and makes it trivially
deterministic.

## Project layout

```
Source_Rasteriser_Triangles/
├── include/
│   └── triangle_rasterizer.h        # stable C ABI (public header)
├── src/
│   ├── raster_internal.h            # internal types & shared helpers
│   ├── raster_common.c              # validation, batch mgmt, public API
│   ├── backend_cpu.c                # CPU (OpenMP scanline) backend
│   └── backend_cuda.cu              # CUDA backend (compiled only with -DCUDA)
├── triangle_rasterizer/             # Python package (ctypes wrapper)
│   ├── __init__.py                  # public API: rasterize, draw_triangle, ...
│   └── _loader.py                   # library discovery + ctypes prototypes
├── tests/
│   ├── test_rasterizer.c            # C self-tests (12 tests)
│   └── python/
│       ├── test_triangle_rasterizer.py  # pytest-style tests (17 tests)
│       └── run_tests.py             # standalone runner (no pytest needed)
├── python/
│   └── demo.py                      # end-to-end demo (renders + writes PNG)
├── examples/
│   ├── common.py                    # shared helpers (load lib, make image)
│   ├── basic_render.py
│   ├── batch_per_frame.py
│   └── cpu_vs_cuda_parity.py        # CPU vs CUDA pixel-identical check
├── scripts/
│   ├── build_cpu.ps1 / build_cpu.sh # configure + build + test (CPU)
│   └── build_cuda.ps1 / build_cuda.sh
├── .github/workflows/ci.yml         # GitHub Actions CI (3 OSes, C + Python)
├── CMakeLists.txt
├── pyproject.toml                   # pip-installable Python package
├── LICENSE                          # MIT
├── .gitignore
└── test_Rasteriser_Triangles.txt    # original pseudocode / spec
```

## Building

The project is built with **CMake**.  A shared library and (optionally) a C
test executable are produced.  The Python package wraps the shared library
via `ctypes`; it finds the library automatically (see
[Installing the Python package](#installing-the-python-package)).

### Windows

> **Prerequisite:** Visual Studio 2019/2022 with the *C++ build tools*
> component (or the *C++ workload*).  Open the **Developer Command Prompt**
> (or "x64 Native Tools Command Prompt") so `cl` is on the `PATH`.

```bat
:: CPU backend (default)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DTRIANGLE_RASTERIZER_BACKEND=AUTO
cmake --build build --config Release

:: or simply run the helper script
scripts\build_cpu.ps1
```

Artifacts land in `build\Release\` (`triangle_rasterizer.dll`,
`triangle_rasterizer.lib`, `triangle_rasterizer_test.exe`).

### Linux / macOS

```sh
# CPU backend (default).  OpenMP is picked up automatically (libgomp / libomp).
cmake -S . -B build -DTRIANGLE_RASTERIZER_BACKEND=AUTO -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# or:
bash scripts/build_cpu.sh
```

Artifacts land in `build/` (`libtriangle_rasterizer.so` /
`.dylib`, `triangle_rasterizer_test`).

> On Linux, ensure `libgomp` is available (it ships with GCC).  On macOS,
> use Apple Clang (or install LLVM) — CMake will find the matching OpenMP
> runtime.

### CUDA backend

Requires the NVIDIA CUDA toolkit (`nvcc` on the `PATH`) and, at run time, a
compatible GPU.

```sh
# Linux
cmake -S . -B build -DTRIANGLE_RASTERIZER_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
# optionally pin architectures, e.g. Ampere + Hopper:
#   -DCMAKE_CUDA_ARCHITECTURES="80;86;90"
cmake --build build --config Release
bash scripts/build_cuda.sh
```

```bat
:: Windows (run from a Developer Command Prompt with nvcc on PATH)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DTRIANGLE_RASTERIZER_BACKEND=CUDA
cmake --build build --config Release
scripts\build_cuda.ps1
```

When the backend is `AUTO`, CMake silently probes for CUDA and falls back to
CPU, so the same command works on machines with or without a GPU.

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `TRIANGLE_RASTERIZER_BACKEND` | `AUTO` | `AUTO`, `CPU`, or `CUDA`. |
| `TRIANGLE_RASTERIZER_TESTS` | `ON` | Build the C test executable. |
| `TRIANGLE_RASTERIZER_OPENMP` | `ON` | Enable OpenMP for the CPU backend. |
| `BUILD_SHARED_LIBS` | `ON` | Build a shared library (recommended). |
| `CMAKE_CUDA_ARCHITECTURES` | `61;70;75;80;86;89;90` | CUDA compute-capability targets. |

## Installing the Python package

The Python package is a thin `ctypes` wrapper; it only needs **NumPy** at
runtime (Pillow is optional, used by the demo/examples to write PNGs).

```sh
# 1) Build the native library first (see Building above).
# 2) Install the package (editable or not):
pip install .
# or, to develop in place:
pip install -e .
```

After installation the package locates the shared library automatically,
searching (in order):

1. the `TRI_RASTERIZER_DLL` environment variable (explicit path),
2. the `build/Release` / `build/Debug` / `build/` directories of the repo,
3. the `triangle_rasterizer/` package directory,
4. the repository root,
5. the system library path (`ctypes.util.find_library`).

If you build the library somewhere else, point the loader at it:

```sh
export TRI_RASTERIZER_DLL=/path/to/libtriangle_rasterizer.so   # Linux/macOS
set TRI_RASTERIZER_DLL=C:\path\to\triangle_rasterizer.dll      # Windows
```

## Using it from Python

```python
import numpy as np
import triangle_rasterizer as tr

# (H, W, 3) or (H, W, 4), uint8, C-contiguous.  Mutated in place.
img = np.zeros((256, 256, 3), dtype=np.uint8)

# vertices are (x, y) in pixel coordinates; colour is 0..255 RGB; alpha 0..1.
triangles = [
    dict(v0=(40, 40),  v1=(216, 40),  v2=(128, 216), color=(255, 0, 0),   alpha=1.0),
    dict(v0=(60, 80),  v1=(196, 80),  v2=(128, 208), color=(0, 200, 255), alpha=0.6),
]

tr.rasterize(img, triangles, n_threads=0)   # 0 -> let the runtime decide

# The two knobs compose freely (defaults: AA_NONE + PRECISION_FLOAT):
tr.rasterize(img, triangles, antialiasing=tr.AA_ANALYTICAL,
             precision=tr.PRECISION_FIXED8)

# Which backend was this library built with?
print(tr.backend())                          # e.g. "cpu (OpenMP)" or "cuda"
```

### Quick API reference

| Call | Description |
|------|-------------|
| `rasterize(img, triangles, n_threads=0, antialiasing=AA_NONE, precision=PRECISION_FLOAT)` | Rasterise a list of triangles into `img` (in place) and return it. |
| `draw_triangle(img, v0, v1, v2, color, alpha=1.0, n_threads=0, antialiasing=AA_NONE, precision=PRECISION_FLOAT)` | Convenience helper for a single triangle. |
| `TriangleBatch(triangles)` | Build a batch handle once, rasterise it repeatedly (`.draw(..., antialiasing=..., precision=...)`). |
| `backend()` | Return the active backend string (`"cpu (OpenMP)"`, `"cpu"`, `"cuda"`). |
| `version()` | Return the library version string. |
| `omp_available()` | `True` if the CPU backend was built with OpenMP. |
| `omp_max_threads()` | OpenMP's current max thread count. |
| `last_error()` / `error_string(code)` | Inspect the last error code / decode a code. |

A triangle is a `dict` (or `RasterTriangle`) with keys: `v0`, `v1`, `v2`
(each an `(x, y)` pair), `color` (an `(r, g, b)` triple in `0..255`), and
`alpha` (optional, `0..1`, default `1.0`).

## Flags at a glance

Every render call takes two **independent** integer knobs. Both default to the
legacy behaviour, so `AA_NONE` + `PRECISION_FLOAT` is byte-for-byte identical to
the original rasteriser. The two compose freely.

| Knob | Value | Constant (Python / C) | Effect |
|------|-------|-----------------------|--------|
| **Anti-aliasing** | `0` | `AA_NONE` / `TRI_AA_NONE` | hard edges, per-pixel inside test (default) |
| | `1` | `AA_ANALYTICAL` / `TRI_AA_ANALYTICAL` | exact per-pixel area coverage (1 tap) — best quality, cheapest AA |
| | `2` | `AA_SSAA2X2` / `TRI_AA_SSAA2X2` | 4 taps on a 2×2 grid — very smooth |
| | `3` | `AA_SSAA4ROT` / `TRI_AA_SSAA4ROT` | 4 taps on a rotated grid — isotropic edges |
| **Precision** | `0` | `PRECISION_FLOAT` / `TRI_PRECISION_FLOAT` | double-precision span math (default, legacy) |
| | `1` | `PRECISION_FIXED8` / `TRI_PRECISION_FIXED8` | Q24.8 integer span math (deterministic, fastest in SSAA) |

### Choosing (combined)

- **Max speed / exact legacy look** → `AA_NONE` (+ `PRECISION_FLOAT` or `FIXED8`).
- **Best quality-per-cost** → `AA_ANALYTICAL` (1 tap, exact straight edges).
- **Smoothest / isotropic edges** → `AA_SSAA2X2` or `AA_SSAA4ROT`.
- **Deterministic cross-platform integer math** → add `PRECISION_FIXED8`
  (matches `PRECISION_FLOAT` within a fraction of a pixel; not bit-identical).

### C entry points (superset chain)

Each feature level adds an entry point that is a strict superset of the last;
calling a higher-level symbol with the lower-level defaults is bit-identical.

| Flags available | Entry point (triangles) | Batch | Single |
|-----------------|--------------------------|-------|--------|
| (none) | `rasterize_triangles` | `rasterize_triangles_batch` | `rasterize_triangle` |
| + AA (`int aa`) | `rasterize_triangles_ex` | `rasterize_triangles_batch_ex` | `rasterize_triangle_ex` |
| + AA + precision (`int aa, int precision`) | `rasterize_triangles_v2` | `rasterize_triangles_batch_v2` | `rasterize_triangle_v2` |

## Anti-aliasing (v1.2.0)

By default the rasteriser uses **hard edges** — every pixel is either fully
covered or not at all (`TRI_AA_NONE`).  v1.2.0 adds optional per-pixel
anti-aliasing so slanted edges get intermediate coverage values blended into
the colour.  It is **off by default**, so every existing caller keeps the exact
legacy behaviour (bit-identical output).

### Modes

| Value | Constant | Method | Cost | Quality |
|-------|----------|--------|------|---------|
| `0` | `AA_NONE` / `TRI_AA_NONE` | none (legacy hard edges) | 1× | — |
| `1` | `AA_ANALYTICAL` / `TRI_AA_ANALYTICAL` | exact geometric per-pixel coverage (1 tap) | ~1.3× | best quality, lowest cost |
| `2` | `AA_SSAA2X2` / `TRI_AA_SSAA2X2` | 2×2 supersampling (4 taps) | ~4× | very smooth |
| `3` | `AA_SSAA4ROT` / `TRI_AA_SSAA4ROT` | 4 rotated supersamples (4 taps) | ~4× | isotropic edges |

- **ANALYTICAL** computes the exact fraction of the pixel that lies inside the
  triangle (the overlap of the pixel's unit square with the triangle's x-span
  at that row).  It is the highest-quality option for the least work, and is
  ideal for flat-colour vector scenes.
- **SSAA2X2 / SSAA4ROT** sample the pixel at 4 sub-pixel taps and count how
  many fall inside the triangle, giving `coverage = hits / 4`.  These are
  classic supersampling filters; `SSAA4ROT` spreads the taps on a diagonal
  (rotated grid) which looks slightly more isotropic on arbitrary angles.

All three AA modes multiply the per-triangle alpha by the computed coverage
(`a_eff = alpha · coverage`) and then use the exact same opaque / alpha-blend
write as the base path, so painter's order and alpha semantics are unchanged.

### Using it (Python)

```python
import triangle_rasterizer as tr
import numpy as np

img = np.zeros((512, 512, 3), dtype=np.uint8)
tri = dict(v0=(60, 80), v1=(440, 80), v2=(256, 460), color=(220, 60, 40), alpha=1.0)

tr.draw_triangle(img, *tri, antialiasing=tr.AA_ANALYTICAL)   # smooth edges
# ...or tr.AA_SSAA2X2, tr.AA_SSAA4ROT, or tr.AA_NONE (default, hard edges)
```

The `antialiasing=` keyword is accepted by `rasterize()`, `draw_triangle()`,
and `TriangleBatch.draw()`.

### Using it (C)

```c
/* The _ex entry points take an extra trailing `int aa` parameter. */
rasterize_triangles_ex(img, W, H, CH, tris, count, /*n_threads=*/0,
                        TRI_AA_ANALYTICAL);
```

The original `rasterize_triangles(...)` / `rasterize_triangle(...)` symbols are
unchanged and always render with `TRI_AA_NONE`.

### Choosing a mode

- **Quality first, low cost** → `AA_ANALYTICAL` (recommended default for vector
  rendering).
- **Classic supersampling look** → `AA_SSAA2X2` or `AA_SSAA4ROT` (4 taps each,
  ~4× the cost of the base path).
- **Maximum speed / exact legacy look** → `AA_NONE`.

See `examples/antialiasing_compare.py` for a side-by-side render of all four
modes with a magnified edge crop.

## Precision (v1.3.0)

The base pipeline computes the per-row triangle x-span and per-pixel coverage
using floating-point (`double`) arithmetic.  v1.3.0 adds an **opt-in fixed-point
path** that performs the span and coverage integer arithmetic in **Q24.8**
format (8 fractional bits, 256 sub-pixel steps) using 64-bit integers, in the
same spirit as the classic `XY_SHIFT`/AET fixed-point fill in `cv::drawContours`.
The float path is unchanged and remains the default.

### Modes

| Value | Constant | Method |
|-------|----------|--------|
| `0` | `PRECISION_FLOAT` / `TRI_PRECISION_FLOAT` | floating-point arithmetic (default, legacy behaviour) |
| `1` | `PRECISION_FIXED8` / `TRI_PRECISION_FIXED8` | Q24.8 fixed-point span + coverage (deterministic integer math) |

- **FIXED8** converts the triangle vertices once up front to fixed-point and
  then does the edge x-span, the min/max span clamping, and the analytical /
  SSAA coverage taps with 64-bit integer arithmetic only.  The result is
  fully deterministic across platforms and thread counts, independent of any
  FPU rounding differences.
- **FIXED8 combines with every AA mode.**  With `AA_NONE` the coverage is a
  hard 0/1; with `AA_ANALYTICAL` / `AA_SSAA*` the fractional coverage taps are
  computed in the same fixed-point domain.

### Using it (Python)

```python
import triangle_rasterizer as tr
import numpy as np

img = np.zeros((512, 512, 3), dtype=np.uint8)
tri = dict(v0=(60, 80), v1=(440, 80), v2=(256, 460), color=(220, 60, 40), alpha=1.0)

tr.draw_triangle(img, *tri, precision=tr.PRECISION_FIXED8)   # fixed-point path
# precision=tr.PRECISION_FLOAT (default) keeps the original float behaviour.
```

The `precision=` keyword is accepted by `rasterize()`, `draw_triangle()`, and
`TriangleBatch.draw()`, and composes with the `antialiasing=` keyword.

### Using it (C)

```c
/* The _v2 entry points take a trailing `int aa` and `int precision`. */
rasterize_triangles_v2(img, W, H, CH, tris, count, /*n_threads=*/0,
                        TRI_AA_ANALYTICAL, TRI_PRECISION_FIXED8);
```

`rasterize_triangles_v2(..., TRI_PRECISION_FLOAT)` is bit-identical to the
v1.2.0 `rasterize_triangles_ex(...)` symbol, and all 1.1.0 / 1.2.0 symbols are
unchanged.

### Accuracy & performance (measured, CPU, 1280×720 scene)

Measured on this machine (see `examples/precision_compare.py`):

- **Accuracy** — `PRECISION_FIXED8` matches the float path within **≤ 1 colour
  level** on only **~0.03 %** of pixels, and those pixels are the analytical-AA
  edge pixels only.  With `AA_NONE` and SSAA modes the two paths are
  **pixel-identical** for the tested scene.
- **Performance** — for the single-tap paths (`AA_NONE`, `AA_ANALYTICAL`)
  `FIXED8` is roughly equal to or slightly faster than float (~0.97–0.99×).
  For the 4-tap SSAA paths it is ~2× slower, because the extra `lround()` per
  sub-pixel tap outweighs the integer-division savings.  Choose `FIXED8` for
  **guaranteed cross-platform bit determinism**, not for speed.

## Using it from C

```c
#include <triangle_rasterizer.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    enum { W = 256, H = 256, CH = 3 };
    unsigned char *img = malloc((size_t)W * H * CH);
    if (!img) return 1;

    RasterTriangle tris[2];
    tris[0].v0[0] = 40;  tris[0].v0[1] = 40;
    tris[0].v1[0] = 216; tris[0].v1[1] = 40;
    tris[0].v2[0] = 128; tris[0].v2[1] = 216;
    tris[0].color[0] = 255; tris[0].color[1] = 0; tris[0].color[2] = 0;
    tris[0].alpha = 1.0f;

    tris[1].v0[0] = 60;  tris[1].v0[1] = 80;
    tris[1].v1[0] = 196; tris[1].v1[1] = 80;
    tris[1].v2[0] = 128; tris[1].v2[1] = 208;
    tris[1].color[0] = 0; tris[1].color[1] = 200; tris[1].color[2] = 255;
    tris[1].alpha = 0.6f;

    int err = rasterize_triangles(img, W, H, CH, tris, 2, 0);   /* n_threads = 0 */
    if (err != TRI_OK) {
        fprintf(stderr, "rasterize failed: %s\n", rasterizer_error_string(err));
        free(img);
        return 1;
    }

    printf("backend: %s\n", rasterizer_backend());
    free(img);
    return 0;
}
```

Compile against the header and link against the shared library.  The C ABI
is additive-only: existing symbols keep their names and signatures across
versions.

## Error codes

| Code | Constant | Meaning |
|------|----------|---------|
| 0 | `TRI_OK` | Success. |
| 1 | `TRI_ERR_NULL` | A required pointer was NULL. |
| 2 | `TRI_ERR_BAD_DIMS` | width/height/channels out of range (channels must be 3 or 4). |
| 3 | `TRI_ERR_EMPTY` | A required count (triangles) is zero. |
| 4 | `TRI_ERR_NOMEM` | Allocation failed. |
| 5 | `TRI_ERR_BAD_VERTEX` | Non-finite vertex coordinate. |
| 6 | `TRI_ERR_NO_THREADS` | OpenMP requested but not available. |
| 7 | `TRI_ERR_BACKEND` | Backend (e.g. CUDA) launch failed. |

`rasterizer_error_string(code)` (C) / `error_string(code)` (Python) returns a
stable, human-readable description.

## Running the tests

There are **two independent test suites** that both exercise the same public
API:

* **C self-tests** (`tests/test_rasterizer.c`, 12 tests) — built by CMake as
  `triangle_rasterizer_test` and run by `ctest`.
* **Python tests** (`tests/python/test_triangle_rasterizer.py`, 17 tests) —
  written in bare `assert` style, so they run under **pytest** *or* the
  bundled standalone runner (no dependencies beyond NumPy).

```sh
# C tests (from the build directory)
cd build && ctest --output-on-failure

# Python tests — standalone (no pytest required):
python tests/python/run_tests.py

# ...or with pytest, if you have it:
python -m pytest tests/python/ -v

# End-to-end demo (writes examples/output/demo.png):
python python/demo.py
```

Both suites verify: known-shape output, alpha blending, RGBA handling,
out-of-bounds clamping, empty-batch and invalid-input error codes, batch
reuse, determinism across thread counts, and (when a GPU is present)
**CPU-vs-CUDA bit-identical parity**.  The parity test **skips gracefully**
when no CUDA device is available.

## Examples

| Script | What it shows |
|--------|---------------|
| `examples/basic_render.py` | Render a handful of triangles and save a PNG. |
| `examples/antialiasing_compare.py` | Render all four AA modes side-by-side with a magnified edge crop. |
| `examples/batch_per_frame.py` | Build one `TriangleBatch`, rasterise it many times (per-frame use). |
| `examples/cpu_vs_cuda_parity.py` | Assert CPU and CUDA outputs are bit-identical (skips without a GPU). |
| `examples/precision_compare.py` | Compare float vs fixed-point (Q24.8) accuracy + performance across all AA modes. |

Each example is runnable directly: `python examples/basic_render.py`.

## Continuous integration

`.github/workflows/ci.yml` builds and tests the project on a matrix of
**Windows, macOS, and Linux** using the CPU backend:

* configure + build with CMake,
* run the C test suite via `ctest`,
* run the Python test suite via the standalone runner,
* run the demo,
* upload the built binaries as artifacts.

A separate job can be enabled on a CUDA-enabled runner to exercise the GPU
backend; the CPU path is the green guarantee on every OS.

## Design notes & limitations

* **Deterministic by construction.** No shared mutable state on the hot path;
  the same input always yields the same pixels, independent of thread count
  and backend.
* **Painter's algorithm.** Triangles are rasterised in input order.  For
  depth-correct ordering of overlapping opaque triangles, sort them yourself
  before the call (e.g. back-to-front for alpha).
* **Colour convention.** RGB `0..255`, alpha `0..1`.  Only the first three
  channels are written; the 4th channel (alpha) of an RGBA buffer is left
  untouched.
* **CUDA is optional.** The project builds and runs without a GPU; CUDA is an
  opt-in backend with automatic CPU fallback at both build time (`AUTO`) and
  run time.
* **No dependencies beyond NumPy** for the Python package (Pillow only for
  writing PNGs in the demo/examples).

## License

MIT — see [`LICENSE`](LICENSE).
