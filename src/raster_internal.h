/*
 * raster_internal.h
 * -----------------------------------------------------------------------------
 * Internal (non-public) declarations shared between the library's common code
 * (batch management, validation, diagnostics, public entry points) and the
 * backend translation unit that actually rasterises (CPU/OpenMP or CUDA).
 *
 * This header is NOT installed and NOT part of the public ABI. Its job is to
 * let the backend reuse the internal triangle record and the shared numeric /
 * geometry helpers without duplicating them, and to expose the single backend
 * entry point the common code calls.
 *
 * Design: the public API (see triangle_rasterizer.h) is implemented ONCE in
 * raster_common.c.  The heavy lifting is delegated to a backend function
 * `tri_backend_rasterize(...)` which is implemented exactly once — by either
 * backend_cpu.c (OpenMP) or backend_cuda.cu (NVIDIA).  CMake links exactly one
 * of those, so the exported ABI is byte-for-byte identical regardless of the
 * backend; only the implementation differs.
 * ---------------------------------------------------------------------------
 */

#ifndef TRIANGLE_RASTERIZER_INTERNAL_H
#define TRIANGLE_RASTERIZER_INTERNAL_H

#include "triangle_rasterizer.h"

#include <math.h>
#include <float.h>
#include <stddef.h>

/* Boundary-inclusion tolerance (pixels, x-direction) for the integer fill range. */
#define TRI_FILL_EPS 1e-6

/* Fixed-point (Q24.8) format used by TRI_PRECISION_FIXED8: 8 fractional bits,
 * i.e. 1/256 px sub-pixel resolution. Mirrors OpenCV's XY_SHIFT technique. */
#define TRI_FX_SHIFT 8
#define TRI_FX_ONE   (1 << TRI_FX_SHIFT)

static inline long tri_to_fixed(double v) {
    return (long)lround(v * (double)TRI_FX_ONE);
}

/* ---------------------------------------------------------------------------
 * Internal triangle record: geometry promoted to double, bbox precomputed,
 * colour kept in [0,255].
 * ------------------------------------------------------------------------- */
typedef struct {
    double v0x, v0y, v1x, v1y, v2x, v2y;
    double cr, cg, cb;     /* colour, each in [0, 255] */
    double alpha;          /* straight alpha in [0, 1] */
    int    minY, maxY;     /* inclusive scanline range (unclamped, from vertices) */
    int    degenerate;     /* 1 if (nearly) zero area -> never drawn */
    /* Q24.8 fixed-point mirrors of v0x..v2y, used only when precision ==
     * TRI_PRECISION_FIXED8 (populated unconditionally at batch build time). */
    long   v0x_fx, v0y_fx, v1x_fx, v1y_fx, v2x_fx, v2y_fx;
} BatchTri;

struct TriangleBatch {
    BatchTri *tri;
    size_t    count;
    size_t    capacity;
};

/* ---------------------------------------------------------------------------
 * Small numeric helpers (shared; `static inline` so each TU inlines its own).
 * ------------------------------------------------------------------------- */
static inline double tri_fmin3(double a, double b, double c) {
    double m = a < b ? a : b;
    return m < c ? m : c;
}
static inline double tri_fmax3(double a, double b, double c) {
    double m = a > b ? a : b;
    return m > c ? m : c;
}
static inline double tri_signed_area2(double ax, double ay, double bx, double by,
                                      double cx, double cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);   /* twice signed area */
}

/* Clamp a [0,255] float to the nearest byte. */
static inline unsigned char tri_to_byte255(float v) {
    if (v < 0.0f)     return 0;
    if (v > 255.0f)   return 255;
    return (unsigned char)(v + 0.5f);
}

/* Blend one channel: out = dst * (1 - a) + src * a, with fast paths. */
static inline unsigned char tri_blend_channel(unsigned char dst, float src, float a) {
    if (a >= 1.0f) return tri_to_byte255(src);
    if (a <= 0.0f) return dst;
    float out = (float)dst * (1.0f - a) + src * a;
    if (out < 0.0f)   out = 0.0f;
    if (out > 255.0f) out = 255.0f;
    return (unsigned char)(out + 0.5f);
}

/* ---------------------------------------------------------------------------
 * Geometry: the closed x-interval of a triangle on scanline `y`.
 * Returns 1 and sets *xmin/*xmax if the triangle intersects the scanline, 0 else.
 * (Implemented identically on CPU and in the CUDA device code for parity.)
 * ------------------------------------------------------------------------- */
static inline int tri_x_span(const BatchTri *t, double y, double *xmin_out, double *xmax_out) {
    double xmin = DBL_MAX, xmax = -DBL_MAX;
    int    hit = 0;

#define TRI_CONSIDER_EDGE(ax, ay, bx, by)                                          \
    do {                                                                           \
        if ((ay <= y && y <= by) || (by <= y && y <= ay)) {                        \
            if (by == ay) {                                                        \
                double lo = (ax) < (bx) ? (ax) : (bx);                             \
                double hi = (ax) < (bx) ? (bx) : (ax);                             \
                if (lo < xmin) xmin = lo;                                          \
                if (hi > xmax) xmax = hi;                                          \
            } else {                                                               \
                double x = (ax) + ((bx) - (ax)) * (y - (ay)) / ((by) - (ay));      \
                if (x < xmin) xmin = x;                                            \
                if (x > xmax) xmax = x;                                            \
            }                                                                      \
            hit = 1;                                                               \
        }                                                                          \
    } while (0)

    TRI_CONSIDER_EDGE(t->v0x, t->v0y, t->v1x, t->v1y);
    TRI_CONSIDER_EDGE(t->v1x, t->v1y, t->v2x, t->v2y);
    TRI_CONSIDER_EDGE(t->v2x, t->v2y, t->v0x, t->v0y);
#undef TRI_CONSIDER_EDGE

    if (!hit) return 0;
    *xmin_out = xmin;
    *xmax_out = xmax;
    return 1;
}

/* ---------------------------------------------------------------------------
 * Fixed-point (Q24.8) counterpart of tri_x_span: same 3-edge scan, but all
 * arithmetic is done in `long` fixed-point (OpenCV-style integer scanline).
 * `y_fx` is the scanline coordinate already in Q24.8. Returns 1 and sets
 * *xmin_fx/*xmax_fx (also Q24.8) if the triangle intersects the scanline.
 * ------------------------------------------------------------------------- */
static inline int tri_x_span_fixed(const BatchTri *t, long y_fx, long *xmin_out, long *xmax_out) {
    long xmin = 0x7fffffffL, xmax = -0x7fffffffL;
    int  hit = 0;

#define TRI_CONSIDER_EDGE_FX(ax, ay, bx, by)                                       \
    do {                                                                           \
        if ((ay <= y_fx && y_fx <= by) || (by <= y_fx && y_fx <= ay)) {            \
            if (by == ay) {                                                        \
                long lo = (ax) < (bx) ? (ax) : (bx);                               \
                long hi = (ax) < (bx) ? (bx) : (ax);                               \
                if (lo < xmin) xmin = lo;                                          \
                if (hi > xmax) xmax = hi;                                          \
            } else {                                                               \
                /* x = ax + (bx-ax) * (y-ay) / (by-ay), kept in Q24.8 throughout   \
                 * via a Q24.8*Q24.8 -> Q24.16 intermediate before the final       \
                 * >> TRI_FX_SHIFT (mirrors OpenCV's shift-based slope maths).     */ \
                long long num = (long long)((bx) - (ax)) * (long long)(y_fx - (ay)); \
                long long den = (long long)((by) - (ay));                          \
                long x = (ax) + (long)(num / den);                                 \
                if (x < xmin) xmin = x;                                            \
                if (x > xmax) xmax = x;                                            \
            }                                                                      \
            hit = 1;                                                               \
        }                                                                           \
    } while (0)

    TRI_CONSIDER_EDGE_FX(t->v0x_fx, t->v0y_fx, t->v1x_fx, t->v1y_fx);
    TRI_CONSIDER_EDGE_FX(t->v1x_fx, t->v1y_fx, t->v2x_fx, t->v2y_fx);
    TRI_CONSIDER_EDGE_FX(t->v2x_fx, t->v2y_fx, t->v0x_fx, t->v0y_fx);
#undef TRI_CONSIDER_EDGE_FX

    if (!hit) return 0;
    *xmin_out = xmin;
    *xmax_out = xmax;
    return 1;
}

/* ---------------------------------------------------------------------------
 * Shared error state (implemented in raster_common.c, used by the backend).
 * C linkage so the CUDA (C++) TU and the C TU agree on the symbol names.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C" {
#endif

void tri_set_error(int code);
int  tri_get_error(void);

/* Backend entry point — implemented exactly once, by backend_cpu.c OR
 * backend_cuda.cu (CMake links one of them).  `aa` is one of the TRI_AA_*
 * modes and `precision` one of the TRI_PRECISION_* modes from the public
 * header; TRI_AA_NONE + TRI_PRECISION_FLOAT is the legacy hard-edge fill and
 * must remain byte-identical to the 1.1.0 behaviour. */
int tri_backend_rasterize(unsigned char *img, int width, int height, int channels,
                          const BatchTri *tri, size_t count, int n_threads, int aa,
                          int precision);

/* Human-readable backend name, e.g. "cpu" or "cuda". */
const char *tri_backend_name(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TRIANGLE_RASTERIZER_INTERNAL_H */
