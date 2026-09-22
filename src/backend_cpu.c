/*
 * backend_cpu.c
 * -----------------------------------------------------------------------------
 * CPU rasterisation backend (parallelised with OpenMP where available).
 *
 * Implements the single backend entry point `tri_backend_rasterize(...)`:
 * one scanline per thread, triangles applied in *input order* within a scanline
 * -> deterministic, data-race-free (each pixel is written by exactly one thread).
 *
 * The exact same algorithm is used whether or not OpenMP is available; with
 * OpenMP the scanlines are simply distributed across threads.
 * ---------------------------------------------------------------------------
 */

#include "raster_internal.h"

#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

const char *tri_backend_name(void) {
#ifdef _OPENMP
    return "cpu (OpenMP)";
#else
    return "cpu";
#endif
}

/* Per-sample x/y offset tables (shared by float + fixed coverage paths). */
static const double tri_off22[4][2] = { {0.25, 0.25}, {0.75, 0.25}, {0.25, 0.75}, {0.75, 0.75} };
static const double tri_off4r[4][2] = { {0.25, 0.50}, {0.50, 0.25}, {0.75, 0.50}, {0.50, 0.75} };

/* Hoist the per-row AA sample x-spans out of the pixel loop.  Computes, for
 * one scanline, the x-span at each AA sample (y+0.5 for analytical, per-tap
 * y for SSAA) exactly once, so the inner pixel loop does cheap compares only.
 * Bit-identical to the old per-pixel tri_x_span recomputation: same values,
 * just derived once per row instead of once per pixel. */
static void tri_row_spans_float(const BatchTri *t, double y, int aa,
                                double tmin[4], double tmax[4], int valid[4]) {
    if (aa == TRI_AA_ANALYTICAL) {
        valid[0] = tri_x_span(t, y + 0.5, &tmin[0], &tmax[0]);
        return;
    }
    const double (*off)[2] = (aa == TRI_AA_SSAA4ROT) ? tri_off4r : tri_off22;
    for (int k = 0; k < 4; k++)
        valid[k] = tri_x_span(t, y + off[k][1], &tmin[k], &tmax[k]);
}

static double tri_px_cov_float(const double tmin[4], const double tmax[4],
                               const int valid[4], int aa, int x) {
    if (aa == TRI_AA_ANALYTICAL) {
        if (!valid[0]) return 0.0;
        double lo = (double)x;
        double hi = lo + 1.0;
        double cov = (hi < tmax[0] ? hi : tmax[0]) - (lo > tmin[0] ? lo : tmin[0]);
        return (cov < 0.0) ? 0.0 : (cov > 1.0) ? 1.0 : cov;
    }
    const double (*off)[2] = (aa == TRI_AA_SSAA4ROT) ? tri_off4r : tri_off22;
    int hits = 0;
    for (int k = 0; k < 4; k++) {
        if (valid[k]) {
            double sx = (double)x + off[k][0];
            if (sx >= tmin[k] && sx <= tmax[k]) hits++;
        }
    }
    return hits * 0.25;
}

/* Fixed-point (Q24.8) counterparts of the two float helpers above. */
static void tri_row_spans_fixed(const BatchTri *t, long y_fx, int aa,
                                long tmin[4], long tmax[4], int valid[4]) {
    if (aa == TRI_AA_ANALYTICAL) {
        valid[0] = tri_x_span_fixed(t, y_fx + (TRI_FX_ONE / 2), &tmin[0], &tmax[0]);
        return;
    }
    const double (*off)[2] = (aa == TRI_AA_SSAA4ROT) ? tri_off4r : tri_off22;
    for (int k = 0; k < 4; k++)
        valid[k] = tri_x_span_fixed(t, y_fx + (long)lround(off[k][1] * TRI_FX_ONE), &tmin[k], &tmax[k]);
}

static double tri_px_cov_fixed(const long tmin[4], const long tmax[4],
                               const int valid[4], int aa, int x) {
    if (aa == TRI_AA_ANALYTICAL) {
        if (!valid[0]) return 0.0;
        long lo_fx = (long)x << TRI_FX_SHIFT;
        long hi_fx = lo_fx + TRI_FX_ONE;
        long cov_fx = (hi_fx < tmax[0] ? hi_fx : tmax[0]) - (lo_fx > tmin[0] ? lo_fx : tmin[0]);
        double cov = (double)cov_fx / (double)TRI_FX_ONE;
        return (cov < 0.0) ? 0.0 : (cov > 1.0) ? 1.0 : cov;
    }
    const double (*off)[2] = (aa == TRI_AA_SSAA4ROT) ? tri_off4r : tri_off22;
    int hits = 0;
    for (int k = 0; k < 4; k++) {
        if (valid[k]) {
            long sx_fx = ((long)x << TRI_FX_SHIFT) + (long)lround(off[k][0] * TRI_FX_ONE);
            if (sx_fx >= tmin[k] && sx_fx <= tmax[k]) hits++;
        }
    }
    return hits * 0.25;
}

/* Exact floor/ceil division by TRI_FX_ONE (a positive power of two), correct
 * for negative numerators too — avoids the epsilon hacks the float path
 * needs, since Q24.8 values are exact rationals. */
static inline long tri_fx_floor_div(long a) {
    long q = a / TRI_FX_ONE;
    if (a % TRI_FX_ONE != 0 && a < 0) q--;
    return q;
}
static inline long tri_fx_ceil_div(long a) {
    long q = a / TRI_FX_ONE;
    if (a % TRI_FX_ONE != 0 && a > 0) q++;
    return q;
}

/* ---------------------------------------------------------------------------
 * Opaque constant-fill span writers (used by the AA_NONE path when the alpha
 * reaches >= 1.0, i.e. a pure fill of the constant RGB triple over the row
 * span).  The output byte values are computed with the EXACT same
 * tri_to_byte255 as the scalar writer, so the result is bit-identical.  The
 * only change is the store width: SSE2 stores 4 pixels per 16-byte lane
 * (RGBA, preserving the untouched 4th alpha byte) or 2 pixels per 8-byte lane
 * (RGB).  The semi-transparent blend path stays on the scalar per-pixel writer
 * because it is genuinely per-channel/per-pixel and would not be bit-identical
 * in SIMD.
 * ------------------------------------------------------------------------- */
#if defined(__SSE2__)
#include <emmintrin.h>
static void span_fill_opaque_rgba(const unsigned char *row, int x_start, int x_end,
                                  unsigned char r, unsigned char g, unsigned char b) {
    int x = x_start;
    while (x + 4 <= x_end + 1) {
        const unsigned char *p = row + (size_t)x * 4;
        __m128i c    = _mm_setr_epi8((int)r, (int)g, (int)b, 0,
                                     (int)r, (int)g, (int)b, 0,
                                     (int)r, (int)g, (int)b, 0,
                                     (int)r, (int)g, (int)b, 0);
        __m128i mask = _mm_set1_epi8(0xff);
        __m128i dst  = _mm_loadu_si128((const __m128i *)p);
        __m128i out  = _mm_or_si128(_mm_and_si128(c, mask),
                                    _mm_andnot_si128(mask, dst));
        _mm_storeu_si128((__m128i *)p, out);
        x += 4;
    }
    for (; x <= x_end; x++) {
        unsigned char *p = (unsigned char *)row + (size_t)x * 4;
        p[0] = r; p[1] = g; p[2] = b;
    }
}

static void span_fill_opaque_rgb(const unsigned char *row, int x_start, int x_end,
                                 unsigned char r, unsigned char g, unsigned char b) {
    int x = x_start;
    while (x + 2 <= x_end + 1) {
        const unsigned char *p = row + (size_t)x * 3;
        __m128i v = _mm_setr_epi8((int)r, (int)g, (int)b, 0,
                                  (int)r, (int)g, (int)b, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0);
        _mm_storel_epi64((__m128i *)p, v);
        x += 2;
    }
    for (; x <= x_end; x++) {
        unsigned char *p = (unsigned char *)row + (size_t)x * 3;
        p[0] = r; p[1] = g; p[2] = b;
    }
}
#endif /* __SSE2__ */

int tri_backend_rasterize(unsigned char *img, int width, int height, int channels,
                          const BatchTri *tri, size_t count, int n_threads, int aa,
                          int precision) {
    (void)height;

    /* ---- Reduce work to the triangle span ---------------------------------- */
    if (count == 0) { tri_set_error(TRI_OK); return TRI_OK; }

    int global_minY = 0x7fffffff, global_maxY = -0x7fffffff;
    for (size_t i = 0; i < count; i++) {
        if (tri[i].degenerate) continue;
        if (tri[i].minY < global_minY) global_minY = tri[i].minY;
        if (tri[i].maxY > global_maxY) global_maxY = tri[i].maxY;
    }
    if (global_maxY < global_minY) { tri_set_error(TRI_OK); return TRI_OK; }
    if (global_minY < 0) global_minY = 0;
    if (global_maxY >= height) global_maxY = height - 1;
    if (global_maxY < global_minY) { tri_set_error(TRI_OK); return TRI_OK; }

    /* ---- Parallel loop (one scanline per iteration) ------------------------ */
    unsigned char *restrict dst = img;
    const int w = width;
    const int ch = channels;
    const int use_fixed = (precision == TRI_PRECISION_FIXED8);
    int y;   /* declared up-front: required by MSVC's OpenMP 2.0 */

#ifdef _OPENMP
    /* MSVC's OpenMP 2.0 has no num_threads(...) clause; set it via API when asked. */
    if (n_threads > 0) omp_set_num_threads(n_threads);
#else
    (void)n_threads;
#endif

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (y = global_minY; y <= global_maxY; y++) {
        for (size_t ti = 0; ti < count; ti++) {
            const BatchTri *t = &tri[ti];
            if (t->degenerate) continue;
            if (t->minY > y || y > t->maxY) continue;          /* outside triangle's y-span */

            int x_start, x_end;
            if (use_fixed) {
                long xmin_fx, xmax_fx;
                long y_fx = (long)y << TRI_FX_SHIFT;
                if (!tri_x_span_fixed(t, y_fx, &xmin_fx, &xmax_fx)) continue;
                x_start = (int)tri_fx_ceil_div(xmin_fx);
                x_end   = (int)tri_fx_floor_div(xmax_fx);
            } else {
                double xmin, xmax;
                if (!tri_x_span(t, (double)y, &xmin, &xmax)) continue;
                x_start = (int)ceil (xmin - TRI_FILL_EPS);
                x_end   = (int)floor(xmax + TRI_FILL_EPS);
            }
            if (x_start < 0)       x_start = 0;
            if (x_end   >= w)      x_end   = w - 1;
            if (x_start > x_end)   continue;

            float r = (float)t->cr, g = (float)t->cg, b = (float)t->cb;
            float a = (float)t->alpha;
            const int do_aa = (aa != TRI_AA_NONE);

            /* Opaque constant-fill fast path (AA_NONE + full alpha): the whole
             * span is a single RGB triple, so write it as wide SIMD lanes.  The
             * byte values come from the identical tri_to_byte255 scalar math, so
             * the output is bit-identical to the per-pixel writer. */
            if (!do_aa && a >= 1.0f) {
                const unsigned char rb = tri_to_byte255(r);
                const unsigned char gb = tri_to_byte255(g);
                const unsigned char bb = tri_to_byte255(b);
                unsigned char *row = dst + (size_t)y * (size_t)w * (size_t)ch;
#if defined(__SSE2__)
                if (ch == 4)      span_fill_opaque_rgba(row, x_start, x_end, rb, gb, bb);
                else if (ch == 3) span_fill_opaque_rgb (row, x_start, x_end, rb, gb, bb);
                else
#endif
                {
                    for (int x = x_start; x <= x_end; x++) {
                        unsigned char *p = row + (size_t)x * (size_t)ch;
                        p[0] = rb; p[1] = gb; p[2] = bb;
                    }
                }
                continue;
            }

            /* Hoist the per-row AA sample x-spans out of the pixel loop so each
             * pixel only does a few compares instead of re-deriving the span. */
            double tmin_f[4], tmax_f[4];
            long   tmin_fx[4], tmax_fx[4];
            int    span_valid[4];
            if (do_aa) {
                if (use_fixed) tri_row_spans_fixed(t, (long)y << TRI_FX_SHIFT, aa, tmin_fx, tmax_fx, span_valid);
                else           tri_row_spans_float(t, (double)y, aa, tmin_f, tmax_f, span_valid);
            }

            for (int x = x_start; x <= x_end; x++) {
                unsigned char *p = dst + (((size_t)y * (size_t)w + (size_t)x) * (size_t)ch);

                float a_eff = a;
                if (do_aa) {
                    float cov = (float)(use_fixed ? tri_px_cov_fixed(tmin_fx, tmax_fx, span_valid, aa, x)
                                                   : tri_px_cov_float(tmin_f, tmax_f, span_valid, aa, x));
                    if (cov <= 0.0f) continue;   /* fully outside under this mode */
                    a_eff = a * cov;             /* modulate alpha by coverage  */
                }

                if (a_eff >= 1.0f) {
                    p[0] = tri_to_byte255(r);
                    p[1] = tri_to_byte255(g);
                    p[2] = tri_to_byte255(b);
                } else if (a_eff > 0.0f) {
                    p[0] = tri_blend_channel(p[0], r, a_eff);
                    p[1] = tri_blend_channel(p[1], g, a_eff);
                    p[2] = tri_blend_channel(p[2], b, a_eff);
                }
            }
        }
    }

    tri_set_error(TRI_OK);
    return TRI_OK;
}
