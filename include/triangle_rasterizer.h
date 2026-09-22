/*
 * triangle_rasterizer.h
 * -----------------------------------------------------------------------------
 * A small, dependency-free 2D triangle rasteriser written in C (C99).
 *
 * Design goals
 *   - Performance & cache-friendly memory layout (single interleaved RGB buffer).
 *   - Deterministic, data-race-free multi-threading via OpenMP:
 *       * a whole triangle is owned by exactly one thread,
 *       * a row inside a triangle is owned by exactly one thread,
 *       * every pixel is written by at most one thread per call,
 *       * overlapping triangles are ordered and never touched concurrently.
 *   - A stable C ABI so it can be compiled to a shared library (.so / .dll /
 *     .dylib) and driven from any foreign-function interface (ctypes, cffi,
 *     pybind11, Rust, ...).
 *
 * Coordinate system
 *   - Image is an RGB(A) buffer, row-major, `channels` bytes per pixel.
 *   - Pixel (x, y): x in [0, width), y in [0, height).
 *   - Flat index of pixel (x, y), channel c:  idx = (y * width + x) * channels + c.
 *   - Vertices are given as (x, y) floats in the same pixel coordinates.
 *   - The triangle is the closed half-plane intersection of its 3 edges, i.e. a
 *     solid triangle inclusive of its boundary (matches the original intent).
 *
 * Alpha blending
 *   - clr_alpha is a straight (non-premultiplied) alpha in [0, 1]:
 *         out = dst * (1 - a) + src * a
 *   - clr_alpha == 1.0 is an opaque fill (fast path, no read of dst).
 *   - clr_alpha == 0.0 is a no-op.
 *
 * Thread safety
 *   - A single call to `rasterize_triangles` is safe to run on its own.
 *   - You may issue *multiple* calls concurrently as long as they do not touch
 *     the same pixels (e.g. disjoint images, or disjoint triangles on one image
 *     where the caller guarantees the ordering). Each call manages its own
 *     internal OpenMP parallelism.
 * ---------------------------------------------------------------------------
 */

#ifndef TRIANGLE_RASTERIZER_H
#define TRIANGLE_RASTERIZER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Export / calling-convention macros (work on MSVC, GCC/Clang, C and C++).
 * ------------------------------------------------------------------------- */
#if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(TRIANGLE_RASTERIZER_BUILDING)
        #define TRIANGLE_RASTERIZER_API __declspec(dllexport)
    #else
        #define TRIANGLE_RASTERIZER_API __declspec(dllimport)
    #endif
#else
    #define TRIANGLE_RASTERIZER_API __attribute__((visibility("default")))
#endif

/* No explicit calling convention is needed: x64 Windows has a single ABI, and
 * on x86 the C default (__cdecl) is already what these functions use. */
#define TRIANGLE_RASTERIZER_CALL

/* Library version. Bump TRIANGLE_RASTERIZER_VERSION on ABI-breaking changes.
 * 1.2.0 — added anti-aliasing enum (TRI_AA_*) and `_ex` rasterize overloads.
 *         All 1.1.0 symbols remain and produce byte-identical output.
 * 1.3.0 — added fixed-point scanline precision enum (TRI_PRECISION_*) and
 *         `_v2` rasterize overloads. All 1.1.0/1.2.0 symbols remain and
 *         produce byte-identical output (they always use TRI_PRECISION_FLOAT). */
#define TRIANGLE_RASTERIZER_VERSION_MAJOR 1
#define TRIANGLE_RASTERIZER_VERSION_MINOR 3
#define TRIANGLE_RASTERIZER_VERSION_PATCH 0
#define TRIANGLE_RASTERIZER_VERSION "1.3.0"

/* ---------------------------------------------------------------------------
 * Opaque handle. A `TriangleBatch` owns a CPU-side array of triangles.
 * The library never mutates vertex data in place, so the handle is safe to
 * share across threads for read-only access.
 * ------------------------------------------------------------------------- */
typedef struct TriangleBatch TriangleBatch;

/* ---------------------------------------------------------------------------
 * A single triangle. Coordinates are floats so sub-pixel (anti-aliased style)
 * positions and non-integer geometry are supported. Order of v0,v1,v2 does not
 * affect the filled region, only the orientation of the edge test; we make the
 * orientation canonical (CCW) internally so the inside test is well-defined.
 * ------------------------------------------------------------------------- */
typedef struct {
    float v0[2];   /* (x, y) */
    float v1[2];   /* (x, y) */
    float v2[2];   /* (x, y) */
    float color[3];/* (r, g, b) each in [0, 255] */
    float alpha;   /* straight alpha in [0, 1] */
} RasterTriangle;

/* ---------------------------------------------------------------------------
 * Return codes. All entry points return one of these (0 == success).
 * ------------------------------------------------------------------------- */
#define TRI_OK                      0
#define TRI_ERR_NULL                1   /* a required pointer was NULL          */
#define TRI_ERR_BAD_DIMS            2   /* width/height/channels out of range   */
#define TRI_ERR_EMPTY               3   /* a required count (triangles) is zero */
#define TRI_ERR_NOMEM               4   /* allocation failed                    */
#define TRI_ERR_BAD_VERTEX          5   /* non-finite vertex coordinate         */
#define TRI_ERR_NO_THREADS          6   /* OpenMP requested but not available   */
#define TRI_ERR_BACKEND             7   /* backend (e.g. CUDA) launch failed    */

/* ---------------------------------------------------------------------------
 * Anti-aliasing modes (v1.2.0, additive; default TRI_AA_NONE is the legacy
 * hard-edge fill and remains byte-identical to the 1.1.0 rasteriser).
 *   - TRI_AA_NONE         legacy per-pixel inside test, hard edges.
 *   - TRI_AA_ANALYTICAL   coverage = area of the pixel covered by the triangle
 *                         on the scanline (exact for straight edges; computed
 *                         from the same closed-form x-span already used).
 *   - TRI_AA_SSAA2X2      4 sub-pixel samples at (1/4,1/4),(3/4,1/4),
 *                         (1/4,3/4),(3/4,3/4); coverage = inside/4.
 *   - TRI_AA_SSAA4ROT     rotated 4-sample grid: (1/4,1/2),(1/2,1/4),
 *                         (3/4,1/2),(1/2,3/4); better for near-axis edges.
 * ------------------------------------------------------------------------- */
enum {
    TRI_AA_NONE       = 0,
    TRI_AA_ANALYTICAL = 1,
    TRI_AA_SSAA2X2    = 2,
    TRI_AA_SSAA4ROT   = 3
};

/* ---------------------------------------------------------------------------
 * Rasterisation precision modes (v1.3.0, additive; default TRI_PRECISION_FLOAT
 * is the legacy double-precision scanline math used by every 1.1.0/1.2.0
 * entry point and is byte-identical to their output).
 *   - TRI_PRECISION_FLOAT   double-precision edge/x-span maths (legacy).
 *   - TRI_PRECISION_FIXED8  Q24.8 fixed-point edge/x-span maths (OpenCV-style
 *                           integer scanline: vertices and slopes quantised to
 *                           1/256 px). Avoids floating-point division on the
 *                           per-row hot path; results match TRI_PRECISION_FLOAT
 *                           to within a fraction of a pixel (not bit-identical).
 * ------------------------------------------------------------------------- */
enum {
    TRI_PRECISION_FLOAT  = 0,
    TRI_PRECISION_FIXED8 = 1
};

/* ---------------------------------------------------------------------------
 * Batch construction / teardown.
 *   - build from an array of RasterTriangle (owned copy is stored), or
 *   - build empty and add triangles one by one (amortised growth).
 * ------------------------------------------------------------------------- */
TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL TriangleBatch *
    rasterizer_batch_new(size_t capacity_hint);

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL TriangleBatch *
    rasterizer_batch_from_triangles(const RasterTriangle *triangles, size_t count);

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterizer_batch_add(TriangleBatch *batch, const RasterTriangle *t);

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL void
    rasterizer_batch_free(TriangleBatch *batch);

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL size_t
    rasterizer_batch_count(const TriangleBatch *batch);

/* ---------------------------------------------------------------------------
 * Rasterisation entry points.
 *
 * img:
 *   - RGB buffer, row-major, `channels` bytes per pixel (3 or 4), little-endian
 *     byte order (R, G, B, [A]). The buffer must already be allocated and at
 *     least `width * height * channels` bytes long.
 *   - `width`, `height` describe the buffer in pixels.
 *   - `channels` is 3 (RGB) or 4 (RGBA). Values outside {3,4} return
 *     TRI_ERR_BAD_DIMS.
 *   - Only the first three channels (RGB) are written.  The 4th channel
 *     (alpha) is left untouched, i.e. the rasterizer composites over the
 *     existing buffer content (standard "over" semantics); it does not
 *     update destination alpha.
 *
 * n_threads:
 *   - 0  -> use the OpenMP default (OMP_NUM_THREADS / hardware concurrency).
 *   - n  -> cap the number of worker threads to n (>= 1).
 *
 * The three overloads all render the SAME image with the SAME triangles; they
 * differ only in how the triangles are supplied.
 * ------------------------------------------------------------------------- */
TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangles(unsigned char *img, int width, int height, int channels,
                        const RasterTriangle *triangles, size_t count,
                        int n_threads);

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangles_batch(unsigned char *img, int width, int height, int channels,
                              const TriangleBatch *batch, int n_threads);

/* Convenience: rasterize a single triangle (thin wrapper). */
TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangle(unsigned char *img, int width, int height, int channels,
                       const RasterTriangle *t, int n_threads);

/* ---------------------------------------------------------------------------
 * Anti-aliasing overloads (v1.2.0, additive).  Same semantics as the 1.1.0
 * functions above plus an explicit TRI_AA_* mode; the 1.1.0 functions are
 * byte-for-byte identical to calling the corresponding `_ex` with TRI_AA_NONE.
 * ------------------------------------------------------------------------- */
TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangles_ex(unsigned char *img, int width, int height, int channels,
                           const RasterTriangle *triangles, size_t count,
                           int n_threads, int aa);

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangles_batch_ex(unsigned char *img, int width, int height, int channels,
                                 const TriangleBatch *batch, int n_threads, int aa);

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangle_ex(unsigned char *img, int width, int height, int channels,
                          const RasterTriangle *t, int n_threads, int aa);

/* ---------------------------------------------------------------------------
 * Precision overloads (v1.3.0, additive).  Same semantics as the `_ex`
 * functions above plus an explicit TRI_PRECISION_* mode; calling `_v2` with
 * TRI_PRECISION_FLOAT is byte-for-byte identical to the corresponding `_ex`.
 * ------------------------------------------------------------------------- */
TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangles_v2(unsigned char *img, int width, int height, int channels,
                           const RasterTriangle *triangles, size_t count,
                           int n_threads, int aa, int precision);

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangles_batch_v2(unsigned char *img, int width, int height, int channels,
                                 const TriangleBatch *batch, int n_threads, int aa,
                                 int precision);

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangle_v2(unsigned char *img, int width, int height, int channels,
                          const RasterTriangle *t, int n_threads, int aa, int precision);

/* ---------------------------------------------------------------------------
 * Diagnostics.
 * ------------------------------------------------------------------------- */
TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterizer_last_error(void);

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL const char *
    rasterizer_error_string(int code);

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL const char *
    rasterizer_version(void);

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterizer_omp_available(void);   /* 1 if OpenMP runtime is linked */
TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterizer_omp_max_threads(void); /* OMP_MAX_ACTIVE_LEVEL-equivalent hint  */

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL const char *
    rasterizer_backend(void);
    /* Stable NUL-terminated backend identifier: "cuda" when the CUDA backend
     * is linked in and a device is present, otherwise "cpu". The pointer is
     * valid for the process lifetime; no copy is required. */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TRIANGLE_RASTERIZER_H */
