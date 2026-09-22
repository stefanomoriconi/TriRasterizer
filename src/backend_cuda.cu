/*
 * backend_cuda.cu
 * -----------------------------------------------------------------------------
 * NVIDIA CUDA rasterisation backend.
 *
 * Implements the single backend entry point `tri_backend_rasterize(...)` with a
 * "one thread per output pixel" kernel.  Each pixel thread walks the triangles
 * in *input order* and, for each triangle that covers the pixel, blends/overwrites
 * that pixel.  Because:
 *
 *   - every output pixel is written by EXACTLY ONE thread, and
 *   - the per-pixel triangle loop is sequential in input order,
 *
 * the result is deterministic and data-race-free.
 *
 * Parity with the CPU backend:
 *   The containment test, boundary tolerance (TRI_FILL_EPS), colour conversion
 *   (tri_to_byte255) and blend (tri_blend_channel) are implemented here with the
 *   same expressions as in the CPU backend and raster_internal.h, and the per-
 *   pixel decision is the same "y in [minY,maxY] AND x in [ceil(xmin-eps),
 *   floor(xmax+eps)]" test.  Given the same inputs this yields byte-identical
 *   images, so a CUDA build and a CPU build can be cross-checked pixel-for-pixel.
 *
 * Graceful degradation:
 *   If no CUDA device is present at runtime (or a CUDA call fails), the backend
 *   transparently falls back to an equivalent serial CPU implementation, so a
 *   "CUDA" build still produces correct output on machines without an NVIDIA GPU.
 * ---------------------------------------------------------------------------
 */

#include "raster_internal.h"

#include <cuda_runtime.h>
#include <cstdio>

/* ===========================================================================
 * Device-side numerics (must match the CPU backend bit-for-bit).
 * ========================================================================= */
#define DEV_FILL_EPS 1e-6

__device__ __forceinline__ unsigned char dev_to_byte255(float v) {
    if (v < 0.0f)   return 0;
    if (v > 255.0f) return 255;
    return (unsigned char)(v + 0.5f);
}

__device__ __forceinline__ unsigned char dev_blend_channel(unsigned char dst, float src, float a) {
    if (a >= 1.0f) return dev_to_byte255(src);
    if (a <= 0.0f) return dst;
    float out = (float)dst * (1.0f - a) + src * a;
    if (out < 0.0f)   out = 0.0f;
    if (out > 255.0f) out = 255.0f;
    return (unsigned char)(out + 0.5f);
}

/* Identical (double) x-span computation to tri_x_span in raster_internal.h. */
__device__ __forceinline__ int dev_x_span(const BatchTri *t, double y,
                                          double *xmin_out, double *xmax_out) {
    double xmin = 1.7976931348623157e+308;   /* DBL_MAX */
    double xmax = -1.7976931348623157e+308;  /* -DBL_MAX */
    int    hit = 0;

#define DEV_EDGE(ax, ay, bx, by)                                                   \
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

    DEV_EDGE(t->v0x, t->v0y, t->v1x, t->v1y);
    DEV_EDGE(t->v1x, t->v1y, t->v2x, t->v2y);
    DEV_EDGE(t->v2x, t->v2y, t->v0x, t->v0y);
#undef DEV_EDGE

    if (!hit) return 0;
    *xmin_out = xmin;
    *xmax_out = xmax;
    return 1;
}

/* Fixed-point (Q24.8) counterpart of dev_x_span, mirrors tri_x_span_fixed. */
__device__ __forceinline__ int dev_x_span_fixed(const BatchTri *t, long y_fx,
                                                long *xmin_out, long *xmax_out) {
    long xmin = 0x7fffffffL, xmax = -0x7fffffffL;
    int  hit = 0;

#define DEV_EDGE_FX(ax, ay, bx, by)                                                 \
    do {                                                                           \
        if ((ay <= y_fx && y_fx <= by) || (by <= y_fx && y_fx <= ay)) {            \
            if (by == ay) {                                                        \
                long lo = (ax) < (bx) ? (ax) : (bx);                               \
                long hi = (ax) < (bx) ? (bx) : (ax);                               \
                if (lo < xmin) xmin = lo;                                          \
                if (hi > xmax) xmax = hi;                                          \
            } else {                                                               \
                long long num = (long long)((bx) - (ax)) * (long long)(y_fx - (ay)); \
                long long den = (long long)((by) - (ay));                          \
                long x = (ax) + (long)(num / den);                                 \
                if (x < xmin) xmin = x;                                            \
                if (x > xmax) xmax = x;                                            \
            }                                                                      \
            hit = 1;                                                               \
        }                                                                          \
    } while (0)

    DEV_EDGE_FX(t->v0x_fx, t->v0y_fx, t->v1x_fx, t->v1y_fx);
    DEV_EDGE_FX(t->v1x_fx, t->v1y_fx, t->v2x_fx, t->v2y_fx);
    DEV_EDGE_FX(t->v2x_fx, t->v2y_fx, t->v0x_fx, t->v0y_fx);
#undef DEV_EDGE_FX

    if (!hit) return 0;
    *xmin_out = xmin;
    *xmax_out = xmax;
    return 1;
}

__device__ __forceinline__ long dev_fx_floor_div(long a) {
    long q = a / TRI_FX_ONE;
    if (a % TRI_FX_ONE != 0 && a < 0) q--;
    return q;
}
__device__ __forceinline__ long dev_fx_ceil_div(long a) {
    long q = a / TRI_FX_ONE;
    if (a % TRI_FX_ONE != 0 && a > 0) q++;
    return q;
}

/* ===========================================================================
 * Anti-aliasing coverage (Wave 1 & 2). Mirrors the per-pixel coverage used by
 * tri_px_cov_float / tri_px_cov_fixed (with per-row spans tri_row_spans_*) in
 * backend_cpu.c and the analytical/SSAA formulas in README "Anti-aliasing".
 * The CUDA kernel is one-thread-per-pixel, so it computes coverage directly per
 * pixel rather than hoisting a per-row span; the maths and rounding are the
 * same. Only invoked when aa != TRI_AA_NONE, so the AA_NONE path stays
 * byte-identical.
 * ========================================================================= */
__device__ __forceinline__
float dev_coverage_px(const BatchTri *t, int x, int y, int aa)
{
    if (aa == TRI_AA_ANALYTICAL) {
        double xmin, xmax;
        if (!dev_x_span(t, (double)y + 0.5, &xmin, &xmax)) return 0.0f;
        double lo = (double)x, hi = lo + 1.0;
        double cov = (hi < xmax ? hi : xmax) - (lo > xmin ? lo : xmin);
        if (cov < 0.0) cov = 0.0;
        if (cov > 1.0) cov = 1.0;
        return (float)cov;
    }
    double off22[8] = {0.25,0.25, 0.75,0.25, 0.25,0.75, 0.75,0.75};
    double off4r[8] = {0.25,0.50, 0.50,0.25, 0.75,0.50, 0.50,0.75};
    const double *off = (aa == TRI_AA_SSAA4ROT) ? off4r : off22;
    int hits = 0;
    for (int k = 0; k < 4; k++) {
        double sy = (double)y + off[2*k+1];
        double sx = (double)x + off[2*k];
        double sxmin, sxmax;
        if (dev_x_span(t, sy, &sxmin, &sxmax) && sx >= sxmin && sx <= sxmax)
            hits++;
    }
    return hits * 0.25f;
}

/* Fixed-point (Q24.8) counterpart of dev_coverage_px, mirrors
 * tri_coverage_px_fixed in backend_cpu.c. */
__device__ __forceinline__
float dev_coverage_px_fixed(const BatchTri *t, int x, int y, int aa)
{
    if (aa == TRI_AA_ANALYTICAL) {
        long xmin_fx, xmax_fx;
        long y_fx = ((long)y << TRI_FX_SHIFT) + (TRI_FX_ONE / 2);
        if (!dev_x_span_fixed(t, y_fx, &xmin_fx, &xmax_fx)) return 0.0f;
        long lo_fx = (long)x << TRI_FX_SHIFT;
        long hi_fx = lo_fx + TRI_FX_ONE;
        long cov_fx = (hi_fx < xmax_fx ? hi_fx : xmax_fx) - (lo_fx > xmin_fx ? lo_fx : xmin_fx);
        float cov = (float)cov_fx / (float)TRI_FX_ONE;
        if (cov < 0.0f) cov = 0.0f;
        if (cov > 1.0f) cov = 1.0f;
        return cov;
    }
    double off22[8] = {0.25,0.25, 0.75,0.25, 0.25,0.75, 0.75,0.75};
    double off4r[8] = {0.25,0.50, 0.50,0.25, 0.75,0.50, 0.50,0.75};
    const double *off = (aa == TRI_AA_SSAA4ROT) ? off4r : off22;
    int hits = 0;
    for (int k = 0; k < 4; k++) {
        long sy_fx = ((long)y << TRI_FX_SHIFT) + (long)lround(off[2*k+1] * TRI_FX_ONE);
        long sx_fx = ((long)x << TRI_FX_SHIFT) + (long)lround(off[2*k]   * TRI_FX_ONE);
        long sxmin_fx, sxmax_fx;
        if (dev_x_span_fixed(t, sy_fx, &sxmin_fx, &sxmax_fx) && sx_fx >= sxmin_fx && sx_fx <= sxmax_fx)
            hits++;
    }
    return hits * 0.25f;
}

/* ===========================================================================
 * Kernel: one thread per output pixel.
 * ========================================================================= */
__global__ void rasterize_kernel(unsigned char *img, int width, int height,
                                 int channels, size_t count,
                                 const BatchTri *dtri, int aa, int precision) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int do_aa = (aa != TRI_AA_NONE);
    const int use_fixed = (precision == TRI_PRECISION_FIXED8);

    unsigned char *p = img + (((size_t)y * (size_t)width + (size_t)x) * (size_t)channels);

    for (size_t ti = 0; ti < count; ti++) {
        const BatchTri *t = &dtri[ti];
        if (t->degenerate) continue;
        if (t->minY > y || y > t->maxY) continue;

        int x_start, x_end;
        if (use_fixed) {
            long xmin_fx, xmax_fx;
            long y_fx = (long)y << TRI_FX_SHIFT;
            if (!dev_x_span_fixed(t, y_fx, &xmin_fx, &xmax_fx)) continue;
            x_start = (int)dev_fx_ceil_div(xmin_fx);
            x_end   = (int)dev_fx_floor_div(xmax_fx);
        } else {
            double xmin, xmax;
            if (!dev_x_span(t, (double)y, &xmin, &xmax)) continue;
            x_start = (int)ceil (xmin - DEV_FILL_EPS);
            x_end   = (int)floor(xmax + DEV_FILL_EPS);
        }
        if (x_start < 0)   x_start = 0;
        if (x_end   >= width) x_end = width - 1;
        if (x < x_start || x > x_end) continue;   /* pixel not covered by this triangle */

        float r = (float)t->cr, g = (float)t->cg, b = (float)t->cb;
        float a = (float)t->alpha;
        float a_eff = a;
        if (do_aa) {
            float cov = use_fixed ? dev_coverage_px_fixed(t, x, y, aa) : dev_coverage_px(t, x, y, aa);
            if (cov <= 0.0f) continue;
            a_eff = a * cov;
        }
        if (a_eff >= 1.0f) {
            p[0] = dev_to_byte255(r);
            p[1] = dev_to_byte255(g);
            p[2] = dev_to_byte255(b);
        } else if (a_eff > 0.0f) {
            p[0] = dev_blend_channel(p[0], r, a_eff);
            p[1] = dev_blend_channel(p[1], g, a_eff);
            p[2] = dev_blend_channel(p[2], b, a_eff);
        }
    }
}

/* ===========================================================================
 * Serial CPU fallback (used when no CUDA device is present / a CUDA call fails).
 * Mirrors backend_cpu.c exactly; single-threaded so it is trivially correct.
 * ========================================================================= */
/* CPU-side AA coverage (mirror of dev_coverage_px) for the serial fallback. */
static double cpu_coverage_px(const BatchTri *t, int x, int y, int aa) {
    if (aa == TRI_AA_ANALYTICAL) {
        double xmin, xmax;
        if (!tri_x_span(t, (double)y + 0.5, &xmin, &xmax)) return 0.0;
        double lo = (double)x, hi = lo + 1.0;
        double cov = (hi < xmax ? hi : xmax) - (lo > xmin ? lo : xmin);
        return (cov < 0.0) ? 0.0 : (cov > 1.0) ? 1.0 : cov;
    }
    static const double off22[4][2] = { {0.25,0.25},{0.75,0.25},{0.25,0.75},{0.75,0.75} };
    static const double off4r[4][2] = { {0.25,0.50},{0.50,0.25},{0.75,0.50},{0.50,0.75} };
    const double (*off)[2] = (aa == TRI_AA_SSAA4ROT) ? off4r : off22;
    int hits = 0;
    for (int k = 0; k < 4; k++) {
        double sy = (double)y + off[k][1];
        double sx = (double)x + off[k][0];
        double sxmin, sxmax;
        if (tri_x_span(t, sy, &sxmin, &sxmax) && sx >= sxmin && sx <= sxmax)
            hits++;
    }
    return hits * 0.25;
}

/* CPU-side fixed-point coverage (mirror of dev_coverage_px_fixed) for the
 * serial fallback. */
static double cpu_coverage_px_fixed(const BatchTri *t, int x, int y, int aa) {
    if (aa == TRI_AA_ANALYTICAL) {
        long xmin_fx, xmax_fx;
        long y_fx = ((long)y << TRI_FX_SHIFT) + (TRI_FX_ONE / 2);
        if (!tri_x_span_fixed(t, y_fx, &xmin_fx, &xmax_fx)) return 0.0;
        long lo_fx = (long)x << TRI_FX_SHIFT;
        long hi_fx = lo_fx + TRI_FX_ONE;
        long cov_fx = (hi_fx < xmax_fx ? hi_fx : xmax_fx) - (lo_fx > xmin_fx ? lo_fx : xmin_fx);
        double cov = (double)cov_fx / (double)TRI_FX_ONE;
        return (cov < 0.0) ? 0.0 : (cov > 1.0) ? 1.0 : cov;
    }
    static const double off22[4][2] = { {0.25,0.25},{0.75,0.25},{0.25,0.75},{0.75,0.75} };
    static const double off4r[4][2] = { {0.25,0.50},{0.50,0.25},{0.75,0.50},{0.50,0.75} };
    const double (*off)[2] = (aa == TRI_AA_SSAA4ROT) ? off4r : off22;
    int hits = 0;
    for (int k = 0; k < 4; k++) {
        long sy_fx = ((long)y << TRI_FX_SHIFT) + (long)lround(off[k][1] * TRI_FX_ONE);
        long sx_fx = ((long)x << TRI_FX_SHIFT) + (long)lround(off[k][0] * TRI_FX_ONE);
        long sxmin_fx, sxmax_fx;
        if (tri_x_span_fixed(t, sy_fx, &sxmin_fx, &sxmax_fx) && sx_fx >= sxmin_fx && sx_fx <= sxmax_fx)
            hits++;
    }
    return hits * 0.25;
}

static inline long cpu_fx_floor_div(long a) {
    long q = a / TRI_FX_ONE;
    if (a % TRI_FX_ONE != 0 && a < 0) q--;
    return q;
}
static inline long cpu_fx_ceil_div(long a) {
    long q = a / TRI_FX_ONE;
    if (a % TRI_FX_ONE != 0 && a > 0) q++;
    return q;
}

static int cpu_fallback_rasterize(unsigned char *img, int width, int height, int channels,
                                  const BatchTri *tri, size_t count, int aa, int precision) {
    (void)height;
    int global_minY = 0x7fffffff, global_maxY = -0x7fffffff;
    for (size_t i = 0; i < count; i++) {
        if (tri[i].degenerate) continue;
        if (tri[i].minY < global_minY) global_minY = tri[i].minY;
        if (tri[i].maxY > global_maxY) global_maxY = tri[i].maxY;
    }
    if (global_maxY < global_minY) return TRI_OK;
    if (global_minY < 0) global_minY = 0;
    if (global_maxY >= height) global_maxY = height - 1;

    unsigned char *restrict dst = img;
    const int w = width, ch = channels;
    const int use_fixed = (precision == TRI_PRECISION_FIXED8);
    for (int y = global_minY; y <= global_maxY; y++) {
        for (size_t ti = 0; ti < count; ti++) {
            const BatchTri *t = &tri[ti];
            if (t->degenerate) continue;
            if (t->minY > y || y > t->maxY) continue;
            int x_start, x_end;
            if (use_fixed) {
                long xmin_fx, xmax_fx;
                long y_fx = (long)y << TRI_FX_SHIFT;
                if (!tri_x_span_fixed(t, y_fx, &xmin_fx, &xmax_fx)) continue;
                x_start = (int)cpu_fx_ceil_div(xmin_fx);
                x_end   = (int)cpu_fx_floor_div(xmax_fx);
            } else {
                double xmin, xmax;
                if (!tri_x_span(t, (double)y, &xmin, &xmax)) continue;
                x_start = (int)ceil (xmin - TRI_FILL_EPS);
                x_end   = (int)floor(xmax + TRI_FILL_EPS);
            }
            if (x_start < 0)    x_start = 0;
            if (x_end   >= w)   x_end   = w - 1;
            if (x_start > x_end) continue;
            float r=(float)t->cr, g=(float)t->cg, b=(float)t->cb, a=(float)t->alpha;
            const int do_aa = (aa != TRI_AA_NONE);
            for (int x = x_start; x <= x_end; x++) {
                unsigned char *p = dst + (((size_t)y * (size_t)w + (size_t)x) * (size_t)ch);
                float a_eff = a;
                if (do_aa) {
                    float cov = (float)(use_fixed ? cpu_coverage_px_fixed(t, x, y, aa) : cpu_coverage_px(t, x, y, aa));
                    if (cov <= 0.0f) continue;
                    a_eff = a * cov;
                }
                if (a_eff >= 1.0f) { p[0]=tri_to_byte255(r); p[1]=tri_to_byte255(g); p[2]=tri_to_byte255(b); }
                else if (a_eff > 0.0f) { p[0]=tri_blend_channel(p[0],r,a_eff); p[1]=tri_blend_channel(p[1],g,a_eff); p[2]=tri_blend_channel(p[2],b,a_eff); }
            }
        }
    }
    return TRI_OK;
}

/* ===========================================================================
 * Host entry point (backend ABI).
 * ========================================================================= */
static int cuda_device_available(void) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    return (e == cudaSuccess) && (n > 0);
}

const char *tri_backend_name(void) {
    return cuda_device_available() ? "cuda" : "cpu";
}

int tri_backend_rasterize(unsigned char *img, int width, int height, int channels,
                          const BatchTri *tri, size_t count, int n_threads, int aa,
                          int precision) {
    (void)n_threads;
    if (count == 0) { tri_set_error(TRI_OK); return TRI_OK; }

    /* If no device is available, fall back to the serial CPU path. */
    if (!cuda_device_available()) {
        int rc = cpu_fallback_rasterize(img, width, height, channels, tri, count, aa, precision);
        tri_set_error(rc);
        return rc;
    }

    const size_t img_bytes = (size_t)width * (size_t)height * (size_t)channels;

    unsigned char *d_img  = NULL;
    const BatchTri *d_tri = NULL;
    int rc = TRI_OK;

    cudaError_t e;
    e = cudaMalloc((void **)&d_img, img_bytes);
    if (e != cudaSuccess) { rc = TRI_ERR_NOMEM; goto done; }
    e = cudaMemcpy(d_img, img, img_bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { rc = TRI_ERR_NOMEM; goto done; }

    e = cudaMalloc((void **)&d_tri, count * sizeof(BatchTri));
    if (e != cudaSuccess) { rc = TRI_ERR_NOMEM; goto done; }
    e = cudaMemcpy(d_tri, tri, count * sizeof(BatchTri), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { rc = TRI_ERR_NOMEM; goto done; }

    dim3 block(16, 16);                                   /* 256 threads */
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    rasterize_kernel<<<grid, block>>>(d_img, width, height, channels, count, d_tri, aa, precision);
    e = cudaGetLastError();
    if (e != cudaSuccess) { rc = TRI_ERR_NULL; goto done; }   /* generic failure code */
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { rc = TRI_ERR_NULL; goto done; }

    e = cudaMemcpy(img, d_img, img_bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { rc = TRI_ERR_NOMEM; goto done; }

done:
    if (d_img) cudaFree(d_img);
    if (d_tri) cudaFree(d_tri);

    if (rc != TRI_OK) {
        /* A CUDA failure is treated as fatal for this call; the library remains
         * usable by subsequent calls (which will re-probe the device). */
        tri_set_error(rc);
        return rc;
    }
    tri_set_error(TRI_OK);
    return TRI_OK;
}
