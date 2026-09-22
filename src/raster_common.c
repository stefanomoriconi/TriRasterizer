/*
 * raster_common.c
 * -----------------------------------------------------------------------------
 * Public C ABI + backend-independent logic:
 *   - batch construction / growth / validation (NaN, degenerate, bbox),
 *   - argument validation and diagnostics,
 *   - the public entry points (rasterize_triangles / _batch / _triangle),
 *   - version, error-string, OpenMP and backend queries.
 *
 * The actual rasterisation is delegated to `tri_backend_rasterize(...)` which
 * is implemented by exactly one of backend_cpu.c / backend_cuda.cu.
 * ---------------------------------------------------------------------------
 */

#include "raster_internal.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* ===========================================================================
 * Error state
 * ========================================================================= */
#if defined(_MSC_VER)
/* MSVC supports __declspec(thread). */
#  define TRI_THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#  define TRI_THREAD_LOCAL _Thread_local
#else
#  define TRI_THREAD_LOCAL static
#endif

static TRI_THREAD_LOCAL int s_last_error = TRI_OK;

void tri_set_error(int code) { s_last_error = code; }
int  tri_get_error(void)     { return s_last_error; }

/* ===========================================================================
 * Validation helpers
 * ========================================================================= */
static int is_finite_float(float v) { return isfinite(v); }

static int validate_triangle(const RasterTriangle *t) {
    if (!t) return 0;
    for (int i = 0; i < 2; i++) {
        if (!is_finite_float(t->v0[i]) || !is_finite_float(t->v1[i]) || !is_finite_float(t->v2[i]))
            return 0;
    }
    return 1;
}

static int validate_dims(int width, int height, int channels) {
    return (width > 0) && (height > 0) && (channels == 3 || channels == 4);
}

static int validate_image(unsigned char *img, int width, int height, int channels) {
    if (!validate_dims(width, height, channels)) return 0;
    if (!img) return 0;
    /* The caller owns the buffer; we only require the declared size to be
     * representable.  (We cannot know the true allocation length from here, so
     * we trust the documented contract: width*height*channels bytes.) */
    return 1;
}

/* ===========================================================================
 * Batch management
 * ========================================================================= */
TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL TriangleBatch *
    rasterizer_batch_new(size_t capacity_hint) {
    TriangleBatch *b = (TriangleBatch *)calloc(1, sizeof(TriangleBatch));
    if (!b) { tri_set_error(TRI_ERR_NOMEM); return NULL; }
    if (capacity_hint > 0) {
        b->capacity = capacity_hint;
        b->tri = (BatchTri *)calloc(capacity_hint, sizeof(BatchTri));
        if (!b->tri) { free(b); tri_set_error(TRI_ERR_NOMEM); return NULL; }
    }
    tri_set_error(TRI_OK);
    return b;
}

/* Push one RasterTriangle into the batch, converting to the internal record. */
static int push_triangle(TriangleBatch *b, const RasterTriangle *t) {
    if (!validate_triangle(t)) { tri_set_error(TRI_ERR_BAD_VERTEX); return 0; }

    if (b->count >= b->capacity) {
        size_t new_cap = (b->capacity == 0) ? 8 : (b->capacity * 2);
        BatchTri *nt = (BatchTri *)realloc(b->tri, new_cap * sizeof(BatchTri));
        if (!nt) { tri_set_error(TRI_ERR_NOMEM); return 0; }
        b->tri = nt;
        b->capacity = new_cap;
    }

    BatchTri *d = &b->tri[b->count];
    d->v0x = (double)t->v0[0]; d->v0y = (double)t->v0[1];
    d->v1x = (double)t->v1[0]; d->v1y = (double)t->v1[1];
    d->v2x = (double)t->v2[0]; d->v2y = (double)t->v2[1];
    d->cr  = (double)t->color[0]; d->cg = (double)t->color[1]; d->cb = (double)t->color[2];
    d->alpha = (double)t->alpha;

    d->v0x_fx = tri_to_fixed(d->v0x); d->v0y_fx = tri_to_fixed(d->v0y);
    d->v1x_fx = tri_to_fixed(d->v1x); d->v1y_fx = tri_to_fixed(d->v1y);
    d->v2x_fx = tri_to_fixed(d->v2x); d->v2y_fx = tri_to_fixed(d->v2y);

    double a2 = tri_signed_area2(d->v0x, d->v0y, d->v1x, d->v1y, d->v2x, d->v2y);
    d->degenerate = (fabs(a2) < 1e-6) ? 1 : 0;

    double ymin = tri_fmin3(d->v0y, d->v1y, d->v2y);
    double ymax = tri_fmax3(d->v0y, d->v1y, d->v2y);
    d->minY = (int)floor(ymin);
    d->maxY = (int)ceil(ymax);

    b->count++;
    return 1;
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL TriangleBatch *
    rasterizer_batch_from_triangles(const RasterTriangle *triangles, size_t count) {
    if (!triangles && count > 0) { tri_set_error(TRI_ERR_NULL); return NULL; }
    TriangleBatch *b = rasterizer_batch_new(count);
    if (!b) return NULL;

    for (size_t i = 0; i < count; i++) {
        if (!push_triangle(b, &triangles[i])) {
            int code = tri_get_error();
            rasterizer_batch_free(b);
            tri_set_error(code);
            return NULL;
        }
    }
    tri_set_error(TRI_OK);
    return b;
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterizer_batch_add(TriangleBatch *batch, const RasterTriangle *t) {
    if (!batch) { tri_set_error(TRI_ERR_NULL); return TRI_ERR_NULL; }
    if (!push_triangle(batch, t)) return tri_get_error();
    tri_set_error(TRI_OK);
    return TRI_OK;
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL void
    rasterizer_batch_free(TriangleBatch *batch) {
    if (!batch) return;
    free(batch->tri);
    free(batch);
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL size_t
    rasterizer_batch_count(const TriangleBatch *batch) {
    return batch ? batch->count : 0;
}

/* ===========================================================================
 * Rasterisation entry points
 * ========================================================================= */
TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangles_batch(unsigned char *img, int width, int height, int channels,
                              const TriangleBatch *batch, int n_threads) {
    if (!batch || batch->count == 0) { tri_set_error(TRI_ERR_EMPTY); return TRI_ERR_EMPTY; }
    if (!validate_image(img, width, height, channels)) {
        int code = (img == NULL) ? TRI_ERR_NULL : TRI_ERR_BAD_DIMS;
        tri_set_error(code); return code;
    }

    int rc = tri_backend_rasterize(img, width, height, channels,
                                   batch->tri, batch->count, n_threads,
                                   TRI_AA_NONE, TRI_PRECISION_FLOAT);
    if (rc != TRI_OK) tri_set_error(rc);
    return rc;
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangles_batch_ex(unsigned char *img, int width, int height, int channels,
                                 const TriangleBatch *batch, int n_threads, int aa) {
    if (!batch || batch->count == 0) { tri_set_error(TRI_ERR_EMPTY); return TRI_ERR_EMPTY; }
    if (!validate_image(img, width, height, channels)) {
        int code = (img == NULL) ? TRI_ERR_NULL : TRI_ERR_BAD_DIMS;
        tri_set_error(code); return code;
    }
    int rc = tri_backend_rasterize(img, width, height, channels,
                                   batch->tri, batch->count, n_threads, aa,
                                   TRI_PRECISION_FLOAT);
    if (rc != TRI_OK) tri_set_error(rc);
    return rc;
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangles_batch_v2(unsigned char *img, int width, int height, int channels,
                                 const TriangleBatch *batch, int n_threads, int aa,
                                 int precision) {
    if (!batch || batch->count == 0) { tri_set_error(TRI_ERR_EMPTY); return TRI_ERR_EMPTY; }
    if (!validate_image(img, width, height, channels)) {
        int code = (img == NULL) ? TRI_ERR_NULL : TRI_ERR_BAD_DIMS;
        tri_set_error(code); return code;
    }
    int rc = tri_backend_rasterize(img, width, height, channels,
                                   batch->tri, batch->count, n_threads, aa, precision);
    if (rc != TRI_OK) tri_set_error(rc);
    return rc;
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangles(unsigned char *img, int width, int height, int channels,
                        const RasterTriangle *triangles, size_t count, int n_threads) {
    if (!triangles && count > 0) { tri_set_error(TRI_ERR_NULL); return TRI_ERR_NULL; }
    if (count == 0)              { tri_set_error(TRI_ERR_EMPTY); return TRI_ERR_EMPTY; }
    if (!validate_image(img, width, height, channels)) {
        int code = (img == NULL) ? TRI_ERR_NULL : TRI_ERR_BAD_DIMS;
        tri_set_error(code); return code;
    }

    /* Build the internal batch (validates every triangle, computes bboxes),
     * then hand it to the backend, then release it. */
    TriangleBatch *b = rasterizer_batch_from_triangles(triangles, count);
    if (!b) return tri_get_error();

    int rc = tri_backend_rasterize(img, width, height, channels, b->tri, b->count, n_threads,
                                   TRI_AA_NONE, TRI_PRECISION_FLOAT);
    rasterizer_batch_free(b);
    if (rc != TRI_OK) tri_set_error(rc);
    return rc;
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangles_ex(unsigned char *img, int width, int height, int channels,
                           const RasterTriangle *triangles, size_t count,
                           int n_threads, int aa) {
    if (!triangles && count > 0) { tri_set_error(TRI_ERR_NULL); return TRI_ERR_NULL; }
    if (count == 0)              { tri_set_error(TRI_ERR_EMPTY); return TRI_ERR_EMPTY; }
    if (!validate_image(img, width, height, channels)) {
        int code = (img == NULL) ? TRI_ERR_NULL : TRI_ERR_BAD_DIMS;
        tri_set_error(code); return code;
    }
    TriangleBatch *b = rasterizer_batch_from_triangles(triangles, count);
    if (!b) return tri_get_error();
    int rc = tri_backend_rasterize(img, width, height, channels, b->tri, b->count, n_threads, aa,
                                   TRI_PRECISION_FLOAT);
    rasterizer_batch_free(b);
    if (rc != TRI_OK) tri_set_error(rc);
    return rc;
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangles_v2(unsigned char *img, int width, int height, int channels,
                           const RasterTriangle *triangles, size_t count,
                           int n_threads, int aa, int precision) {
    if (!triangles && count > 0) { tri_set_error(TRI_ERR_NULL); return TRI_ERR_NULL; }
    if (count == 0)              { tri_set_error(TRI_ERR_EMPTY); return TRI_ERR_EMPTY; }
    if (!validate_image(img, width, height, channels)) {
        int code = (img == NULL) ? TRI_ERR_NULL : TRI_ERR_BAD_DIMS;
        tri_set_error(code); return code;
    }
    TriangleBatch *b = rasterizer_batch_from_triangles(triangles, count);
    if (!b) return tri_get_error();
    int rc = tri_backend_rasterize(img, width, height, channels, b->tri, b->count, n_threads, aa, precision);
    rasterizer_batch_free(b);
    if (rc != TRI_OK) tri_set_error(rc);
    return rc;
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangle(unsigned char *img, int width, int height, int channels,
                       const RasterTriangle *t, int n_threads) {
    if (!t) { tri_set_error(TRI_ERR_NULL); return TRI_ERR_NULL; }
    return rasterize_triangles(img, width, height, channels, t, 1, n_threads);
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangle_ex(unsigned char *img, int width, int height, int channels,
                          const RasterTriangle *t, int n_threads, int aa) {
    if (!t) { tri_set_error(TRI_ERR_NULL); return TRI_ERR_NULL; }
    return rasterize_triangles_ex(img, width, height, channels, t, 1, n_threads, aa);
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int
    rasterize_triangle_v2(unsigned char *img, int width, int height, int channels,
                          const RasterTriangle *t, int n_threads, int aa, int precision) {
    if (!t) { tri_set_error(TRI_ERR_NULL); return TRI_ERR_NULL; }
    return rasterize_triangles_v2(img, width, height, channels, t, 1, n_threads, aa, precision);
}

/* ===========================================================================
 * Diagnostics
 * ========================================================================= */
TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int rasterizer_last_error(void) {
    return tri_get_error();
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL const char *rasterizer_error_string(int code) {
    switch (code) {
        case TRI_OK:                 return "OK";
        case TRI_ERR_NULL:           return "NULL argument";
        case TRI_ERR_BAD_DIMS:       return "invalid width/height/channels";
        case TRI_ERR_EMPTY:          return "no triangles to rasterise";
        case TRI_ERR_NOMEM:          return "out of memory";
        case TRI_ERR_BAD_VERTEX:     return "non-finite vertex coordinate";
        case TRI_ERR_NO_THREADS:     return "OpenMP requested but not available";
        case TRI_ERR_BACKEND:        return "rasterisation backend failed";
        default:                     return "unknown error";
    }
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL const char *rasterizer_version(void) {
    return TRIANGLE_RASTERIZER_VERSION;
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int rasterizer_omp_available(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL int rasterizer_omp_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

TRIANGLE_RASTERIZER_API TRIANGLE_RASTERIZER_CALL const char *rasterizer_backend(void) {
    return tri_backend_name();
}
