/*
 * test_rasterizer.c
 * -----------------------------------------------------------------------------
 * Self-test for the triangle rasteriser. Run as:  triangle_rasterizer_test
 *
 * Checks
 *   1. Opaque triangle fills an interior pixel to exactly the source colour and
 *      leaves an exterior pixel untouched.
 *   2. Two triangles: the top one wins where they overlap (painter's order).
 *   3. Determinism: 1 vs 8 threads produce a bit-identical image.
 *   4. Alpha blend: dst=(0,0,0), src=(255,0,0), a=0.5 -> (128,0,0).
 *   5. RGBA: the A byte is preserved by the rasteriser.
 *   6. Degenerate (zero-area) triangle draws nothing.
 *   7. Error paths: NULL image, bad channel count, non-finite vertex.
 * ---------------------------------------------------------------------------
 */
#include "triangle_rasterizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Image dimensions as constant expressions (required for array sizing in C). */
#define TEST_W 128
#define TEST_H 128
#define TEST_CH 3

static int g_failures = 0;

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "[FAIL] %s:%d  %s\n", __FILE__, __LINE__, (msg));   \
            g_failures++;                                                        \
        }                                                                        \
    } while (0)

static int buf_eq(const unsigned char *a, const unsigned char *b, size_t n) {
    return memcmp(a, b, n) == 0;
}

static int pixel_eq(const unsigned char *img, int w, int h, int ch,
                    int x, int y, unsigned char r, unsigned char g, unsigned char b) {
    const size_t i = ((size_t)y * (size_t)w + (size_t)x) * (size_t)ch;
    return img[i] == r && img[i + 1] == g && img[i + 2] == b;
}

/* --------------------------------------------------------------------------- */
int main(void) {
    printf("triangle_rasterizer  version=%s  omp=%d max_threads=%d\n",
           rasterizer_version(),
           rasterizer_omp_available(),
           rasterizer_omp_max_threads());

    const int W = TEST_W, H = TEST_H, CH = TEST_CH;
    unsigned char imgA[TEST_W * TEST_H * TEST_CH];
    unsigned char imgB[TEST_W * TEST_H * TEST_CH];
    unsigned char imgC[TEST_W * TEST_H * TEST_CH];
    memset(imgA, 0, sizeof(imgA));
    memset(imgB, 0, sizeof(imgB));
    memset(imgC, 0, sizeof(imgC));

    /* ---- test 1: opaque triangle -------------------------------------- */
    {
        RasterTriangle t = {
            .v0   = { 64.0f,  32.0f },
            .v1   = {  32.0f, 96.0f },
            .v2   = {  96.0f, 96.0f },
            .color= { 200.0f, 100.0f,  50.0f },
            .alpha= 1.0f,
        };
        int rc = rasterize_triangle(imgA, W, H, CH, &t, 0);
        CHECK(rc == TRI_OK, "opaque: call succeeded");

        /* Centroid is inside. */
        CHECK(pixel_eq(imgA, W, H, CH, 64, 74, 200, 100, 50),
              "opaque: centroid pixel has exact fill colour");
        /* A corner pixel is outside. */
        CHECK(pixel_eq(imgA, W, H, CH, 0, 0, 0, 0, 0),
              "opaque: corner pixel is untouched");
        /* Just outside the triangle's bottom-left. */
        CHECK(pixel_eq(imgA, W, H, CH, 20, 96, 0, 0, 0),
              "opaque: a just-outside pixel is untouched");
    }

    /* ---- test 2: painter's order (overlap) ---------------------------- */
    {
        RasterTriangle a = {
            .v0 = { 20.0f, 20.0f }, .v1 = { 80.0f, 20.0f }, .v2 = { 50.0f, 80.0f },
            .color = { 255.0f, 0.0f, 0.0f }, .alpha = 1.0f,
        };
        RasterTriangle b = {
            .v0 = { 50.0f, 50.0f }, .v1 = { 110.0f, 50.0f }, .v2 = { 80.0f, 110.0f },
            .color = { 0.0f, 255.0f, 0.0f }, .alpha = 1.0f,
        };
        RasterTriangle tris[2] = { a, b };
        int rc = rasterize_triangles(imgB, W, H, CH, tris, 2, 0);
        CHECK(rc == TRI_OK, "overlap: call succeeded");

        /* (45,60) is inside 'a' only -> red.  */
        CHECK(pixel_eq(imgB, W, H, CH, 45, 60, 255, 0, 0),
              "overlap: pixel in lower triangle is red");
        /* (90,60) is inside 'b' only -> green.  */
        CHECK(pixel_eq(imgB, W, H, CH, 90, 60, 0, 255, 0),
              "overlap: pixel in upper triangle is green");
        /* (57,60) is inside BOTH -> b (drawn later) wins -> green. */
        CHECK(pixel_eq(imgB, W, H, CH, 57, 60, 0, 255, 0),
              "overlap: overlapping pixel shows the later (upper) triangle");
    }

    /* ---- test 3: determinism across thread counts ---------------------- */
    {
        /* A random-looking batch of 64 triangles, drawn into two images with
         * different thread counts; they MUST be bit-identical. */
        srand(1234);
        const size_t N = 64;
        RasterTriangle *tri = (RasterTriangle *)malloc(N * sizeof(RasterTriangle));
        for (size_t i = 0; i < N; i++) {
            tri[i].v0[0] = (float)rand() / RAND_MAX * (W - 1) * 1.0f;
            tri[i].v0[1] = (float)rand() / RAND_MAX * (H - 1) * 1.0f;
            tri[i].v1[0] = (float)rand() / RAND_MAX * (W - 1) * 1.0f;
            tri[i].v1[1] = (float)rand() / RAND_MAX * (H - 1) * 1.0f;
            tri[i].v2[0] = (float)rand() / RAND_MAX * (W - 1) * 1.0f;
            tri[i].v2[1] = (float)rand() / RAND_MAX * (H - 1) * 1.0f;
            tri[i].color[0] = (float)rand();
            tri[i].color[1] = (float)rand();
            tri[i].color[2] = (float)rand();
            tri[i].alpha    = 0.2f + 0.8f * ((float)rand() / RAND_MAX);
        }

        unsigned char imgT1[TEST_W * TEST_H * TEST_CH];
        unsigned char imgT2[TEST_W * TEST_H * TEST_CH];
        memset(imgT1, 200, sizeof(imgT1));   /* non-zero dst to exercise blending */
        memcpy(imgT2, imgT1, sizeof(imgT1));

        CHECK(rasterize_triangles(imgT1, W, H, CH, tri, N, 1) == TRI_OK, "det-1t");
        CHECK(rasterize_triangles(imgT2, W, H, CH, tri, N, 8) == TRI_OK, "det-8t");
        CHECK(buf_eq(imgT1, imgT2, W * H * CH),
              "determinism: 1 thread == 8 threads (bit-identical)");

        free(tri);
    }

    /* ---- test 4: alpha blend ------------------------------------------ */
    {
        RasterTriangle t = {
            .v0 = { 0.0f, 0.0f }, .v1 = { 100.0f, 0.0f }, .v2 = { 0.0f, 100.0f },
            .color = { 255.0f, 0.0f, 0.0f }, .alpha = 0.5f,
        };
        /* Start with a black background. */
        memset(imgC, 0, sizeof(imgC));
        CHECK(rasterize_triangle(imgC, W, H, CH, &t, 0) == TRI_OK, "alpha: call ok");

        /* At (50,50), the triangle covers the pixel; 0.5 blend over black =>
         * 255 * 0.5 = 127.5 -> rounds to 128. */
        CHECK(pixel_eq(imgC, W, H, CH, 50, 50, 128, 0, 0),
              "alpha: 0.5 blend of pure red over black is (128,0,0)");
        /* A pixel outside the triangle stays black. */
        CHECK(pixel_eq(imgC, W, H, CH, 127, 127, 0, 0, 0),
              "alpha: exterior pixel unchanged");
    }

    /* ---- test 5: RGBA A-byte is preserved ------------------------------ */
    {
        const int C4 = 4;
        unsigned char imgR[TEST_W * TEST_H * 4];
        for (size_t i = 0; i < (size_t)W * H * C4; i += C4) imgR[i + 3] = 99;
        /* Pre-set the R,G,B channels to non-zero so blending is non-trivial. */
        for (size_t i = 0; i < (size_t)W * H; i++) {
            imgR[i * C4 + 0] = 10; imgR[i * C4 + 1] = 20; imgR[i * C4 + 2] = 30;
        }

        RasterTriangle t = {
            .v0 = { 0.0f, 0.0f }, .v1 = { 100.0f, 0.0f }, .v2 = { 0.0f, 100.0f },
            .color = { 255.0f, 255.0f, 255.0f }, .alpha = 1.0f,
        };
        CHECK(rasterize_triangle(imgR, W, H, C4, &t, 0) == TRI_OK, "rgba: call ok");
        /* Opaque fill changes RGB to (255,255,255) but leaves the A byte (99). */
        CHECK(pixel_eq(imgR, W, H, C4, 20, 20, 255, 255, 255),
              "rgba: RGB channels are filled");
        CHECK(imgR[((size_t)20 * W + 20) * C4 + 3] == 99,
              "rgba: the A byte is preserved");
    }

    /* ---- test 6: degenerate (zero-area) triangle ----------------------- */
    {
        unsigned char imgD[TEST_W * TEST_H * TEST_CH];
        memset(imgD, 10, sizeof(imgD));
        RasterTriangle t = {
            .v0 = { 0.0f, 0.0f }, .v1 = { 10.0f, 0.0f }, .v2 = { 5.0f, 0.0f },
            .color = { 255.0f, 255.0f, 255.0f }, .alpha = 1.0f,
        };
        CHECK(rasterize_triangle(imgD, W, H, CH, &t, 0) == TRI_OK, "degenerate: ok");
        /* Every pixel unchanged (10,10,10). */
        int untouched = 1;
        for (size_t i = 0; i < (size_t)W * H * CH; i++) if (imgD[i] != 10) { untouched = 0; break; }
        CHECK(untouched, "degenerate: no pixels were modified");
    }

    /* ---- test 7: error paths ------------------------------------------ */
    {
        RasterTriangle t = {
            .v0 = { 0.0f, 0.0f }, .v1 = { 10.0f, 0.0f }, .v2 = { 0.0f, 10.0f },
            .color = { 0.0f, 0.0f, 0.0f }, .alpha = 1.0f,
        };
        CHECK(rasterize_triangle(NULL, W, H, CH, &t, 0) == TRI_ERR_NULL, "err: NULL image");
        CHECK(rasterize_triangles(imgA, W, H, 5, &t, 1, 0) == TRI_ERR_BAD_DIMS,
              "err: bad channels");
        t.v0[0] = NAN;
        CHECK(rasterize_triangle(imgA, W, H, CH, &t, 0) == TRI_ERR_BAD_VERTEX,
              "err: NaN vertex");
    }

    /* ---- batch API ------------------------------------------------------ */
    {
        TriangleBatch *b = rasterizer_batch_new(0);
        CHECK(b != NULL, "batch: created");
        RasterTriangle t = {
            .v0 = { 10.0f, 10.0f }, .v1 = { 110.0f, 10.0f }, .v2 = { 60.0f, 110.0f },
            .color = { 1.0f, 2.0f, 3.0f }, .alpha = 1.0f,
        };
        CHECK(rasterizer_batch_add(b, &t) == TRI_OK, "batch: add");
        CHECK(rasterizer_batch_count(b) == 1, "batch: count == 1");

        unsigned char imgE[TEST_W * TEST_H * TEST_CH];
        memset(imgE, 0, sizeof(imgE));
        CHECK(rasterize_triangles_batch(imgE, W, H, CH, b, 0) == TRI_OK, "batch: rasterize");
        /* Centroid: (10+110+60)/3, (10+10+110)/3 = 60, 43.33 -> inside. */
        CHECK(pixel_eq(imgE, W, H, CH, 60, 43, 1, 2, 3),
              "batch: centroid pixel is filled");
        rasterizer_batch_free(b);
    }

    /* ---- test 8: anti-aliasing (v1.2.0) --------------------------------- */
    {
        /* Right triangle with a horizontal top edge at y=0, a vertical left
         * edge at x=0, and the hypotenuse from (100,0) to (0,100).  At a given
         * row y the x-span runs from 0 to 100-y.  This gives us precise,
         * analytically known coverage on the hypotenuse. */
        RasterTriangle t = {
            .v0 = { 0.0f, 0.0f }, .v1 = { 100.0f, 0.0f }, .v2 = { 0.0f, 100.0f },
            .color = { 255.0f, 0.0f, 0.0f }, .alpha = 1.0f,
        };

        unsigned char imgN[TEST_W * TEST_H * TEST_CH];
        unsigned char imgAa[TEST_W * TEST_H * TEST_CH];
        memset(imgN, 0, sizeof(imgN));
        memset(imgAa, 0, sizeof(imgAa));

        /* 8a. ANALYTICAL: an interior pixel is fully covered -> full red. */
        CHECK(rasterize_triangle_ex(imgAa, W, H, CH, &t, 0, TRI_AA_ANALYTICAL) == TRI_OK,
              "aa: analytical call ok");
        CHECK(pixel_eq(imgAa, W, H, CH, 10, 10, 255, 0, 0),
              "aa: analytical interior pixel is fully filled");

        /* 8b. ANALYTICAL: boundary pixel on the hypotenuse is PARTIALLY
         * covered.  At row y=50 the hypotenuse is at x=50.  Pixel x=50 spans
         * [50,51]; coverage = min(51,50)-max(50,0) = 0  (just outside), while
         * pixel x=49 spans [49,50]; coverage = min(51,50)-49 = 1 -> wait: the
         * x-span at y+0.5=50.5 is [0, 49.5].  So pixel 49 spans [49,50],
         * coverage = min(50,49.5)-49 = 0.5 -> 255*0.5 = 127.5 -> 128. */
        {
            int r = (int)imgAa[((size_t)50 * W + 49) * CH + 0];
            CHECK(r > 0 && r < 255,
                  "aa: analytical hypotenuse pixel has partial (0,255) coverage");
        }

        /* 8c. SSAA modes: interior full, exterior zero. */
        unsigned char imgS2[TEST_W * TEST_H * TEST_CH];
        unsigned char imgS4[TEST_W * TEST_H * TEST_CH];
        memset(imgS2, 0, sizeof(imgS2));
        memset(imgS4, 0, sizeof(imgS4));
        CHECK(rasterize_triangle_ex(imgS2, W, H, CH, &t, 0, TRI_AA_SSAA2X2) == TRI_OK,
              "aa: ssaa2x2 call ok");
        CHECK(rasterize_triangle_ex(imgS4, W, H, CH, &t, 0, TRI_AA_SSAA4ROT) == TRI_OK,
              "aa: ssaa4rot call ok");
        CHECK(pixel_eq(imgS2, W, H, CH, 10, 10, 255, 0, 0),
              "aa: ssaa2x2 interior pixel is fully filled");
        CHECK(pixel_eq(imgS4, W, H, CH, 10, 10, 255, 0, 0),
              "aa: ssaa4rot interior pixel is fully filled");
        CHECK(pixel_eq(imgS2, W, H, CH, 120, 120, 0, 0, 0),
              "aa: ssaa2x2 exterior pixel untouched");

        /* 8d. SSAA and ANALYTICAL produce similar (non-identical) coverage on
         * the hypotenuse: both must show partial coverage at pixel (49,50). */
        {
            int r2 = (int)imgS2[((size_t)50 * W + 49) * CH + 0];
            int r4 = (int)imgS4[((size_t)50 * W + 49) * CH + 0];
            CHECK(r2 > 0 && r2 < 255, "aa: ssaa2x2 hypotenuse pixel is partial");
            CHECK(r4 > 0 && r4 < 255, "aa: ssaa4rot hypotenuse pixel is partial");
        }

        /* 8e. AA_NONE via the _ex entry point is bit-identical to the legacy
         * entry point (regression guard). */
        unsigned char imgLegacy[TEST_W * TEST_H * TEST_CH];
        unsigned char imgExNone[TEST_W * TEST_H * TEST_CH];
        memset(imgLegacy, 7, sizeof(imgLegacy));
        memset(imgExNone, 7, sizeof(imgExNone));
        CHECK(rasterize_triangle(imgLegacy, W, H, CH, &t, 0) == TRI_OK, "aa: legacy ok");
        CHECK(rasterize_triangle_ex(imgExNone, W, H, CH, &t, 0, TRI_AA_NONE) == TRI_OK,
              "aa: ex-none ok");
        CHECK(buf_eq(imgLegacy, imgExNone, W * H * CH),
              "aa: TRI_AA_NONE (_ex) == legacy rasterize_triangle (bit-identical)");

        /* 8f. Determinism across thread counts for an AA mode. */
        unsigned char imgD1[TEST_W * TEST_H * TEST_CH];
        unsigned char imgD8[TEST_W * TEST_H * TEST_CH];
        memset(imgD1, 0, sizeof(imgD1));
        memset(imgD8, 0, sizeof(imgD8));
        CHECK(rasterize_triangles_ex(imgD1, W, H, CH, &t, 1, 1, TRI_AA_ANALYTICAL) == TRI_OK,
              "aa: det 1t ok");
        CHECK(rasterize_triangles_ex(imgD8, W, H, CH, &t, 1, 8, TRI_AA_ANALYTICAL) == TRI_OK,
              "aa: det 8t ok");
        CHECK(buf_eq(imgD1, imgD8, W * H * CH),
              "aa: analytical 1 thread == 8 threads (bit-identical)");

        /* 8g. Painter's order is preserved under AA: later triangle wins. */
        RasterTriangle a = {
            .v0 = { 20.0f, 20.0f }, .v1 = { 80.0f, 20.0f }, .v2 = { 50.0f, 80.0f },
            .color = { 255.0f, 0.0f, 0.0f }, .alpha = 1.0f,
        };
        RasterTriangle bb = {
            .v0 = { 50.0f, 50.0f }, .v1 = { 110.0f, 50.0f }, .v2 = { 80.0f, 110.0f },
            .color = { 0.0f, 255.0f, 0.0f }, .alpha = 1.0f,
        };
        RasterTriangle tris[2] = { a, bb };
        unsigned char imgP[TEST_W * TEST_H * TEST_CH];
        memset(imgP, 0, sizeof(imgP));
        CHECK(rasterize_triangles_ex(imgP, W, H, CH, tris, 2, 0, TRI_AA_ANALYTICAL) == TRI_OK,
              "aa: painter call ok");
        /* (57,60) is inside both; the later (green) triangle must win. */
        CHECK(pixel_eq(imgP, W, H, CH, 57, 60, 0, 255, 0),
              "aa: painter's order preserved under ANALYTICAL");
    }

    /* ---- test 9: fixed-point precision (v1.3.0) -------------------------- */
    {
        RasterTriangle t = {
            .v0 = { 5.3f, 4.2f }, .v1 = { 100.7f, 10.1f }, .v2 = { 30.0f, 95.9f },
            .color = { 200.0f, 50.0f, 10.0f }, .alpha = 1.0f,
        };

        /* 9a. PRECISION_FLOAT via `_v2` is bit-identical to `_ex`. */
        unsigned char imgEx[TEST_W * TEST_H * TEST_CH];
        unsigned char imgV2f[TEST_W * TEST_H * TEST_CH];
        memset(imgEx, 0, sizeof(imgEx));
        memset(imgV2f, 0, sizeof(imgV2f));
        CHECK(rasterize_triangle_ex(imgEx, W, H, CH, &t, 0, TRI_AA_NONE) == TRI_OK,
              "precision: ex ok");
        CHECK(rasterize_triangle_v2(imgV2f, W, H, CH, &t, 0, TRI_AA_NONE, TRI_PRECISION_FLOAT) == TRI_OK,
              "precision: v2/float ok");
        CHECK(buf_eq(imgEx, imgV2f, W * H * CH),
              "precision: v2 with TRI_PRECISION_FLOAT == _ex (bit-identical)");

        /* 9b. PRECISION_FIXED8 hard-edge fill matches the float path closely:
         * every differing pixel must be a single-pixel boundary rounding
         * case (both non-zero would mean a colour mismatch, which cannot
         * happen here since only one triangle is drawn). */
        unsigned char imgFx[TEST_W * TEST_H * TEST_CH];
        memset(imgFx, 0, sizeof(imgFx));
        CHECK(rasterize_triangle_v2(imgFx, W, H, CH, &t, 0, TRI_AA_NONE, TRI_PRECISION_FIXED8) == TRI_OK,
              "precision: v2/fixed8 ok");
        {
            size_t ndiff = 0;
            for (size_t i = 0; i < (size_t)W * H; i++) {
                if (imgEx[i * CH] != imgFx[i * CH]) ndiff++;
            }
            /* Sub-pixel accurate to 1/256px: only a handful of edge pixels
             * (if any) may disagree, never a large fraction of the triangle. */
            CHECK(ndiff < 30, "precision: fixed8 hard-edge closely matches float (few boundary px differ)");
        }

        /* 9c. PRECISION_FIXED8 + TRI_AA_ANALYTICAL: coverage differs from the
         * float path by at most one 8-bit colour level (rounding only). */
        unsigned char imgFa[TEST_W * TEST_H * TEST_CH];
        unsigned char imgFxA[TEST_W * TEST_H * TEST_CH];
        memset(imgFa, 0, sizeof(imgFa));
        memset(imgFxA, 0, sizeof(imgFxA));
        CHECK(rasterize_triangle_v2(imgFa, W, H, CH, &t, 0, TRI_AA_ANALYTICAL, TRI_PRECISION_FLOAT) == TRI_OK,
              "precision: v2/float+aa ok");
        CHECK(rasterize_triangle_v2(imgFxA, W, H, CH, &t, 0, TRI_AA_ANALYTICAL, TRI_PRECISION_FIXED8) == TRI_OK,
              "precision: v2/fixed8+aa ok");
        {
            int max_diff = 0;
            for (size_t i = 0; i < (size_t)W * H * CH; i++) {
                int d = (int)imgFa[i] - (int)imgFxA[i];
                if (d < 0) d = -d;
                if (d > max_diff) max_diff = d;
            }
            CHECK(max_diff <= 1, "precision: fixed8 analytical AA within 1 colour level of float");
        }

        /* 9d. Determinism across thread counts for the fixed-point path. */
        unsigned char imgD1[TEST_W * TEST_H * TEST_CH];
        unsigned char imgD8[TEST_W * TEST_H * TEST_CH];
        memset(imgD1, 0, sizeof(imgD1));
        memset(imgD8, 0, sizeof(imgD8));
        CHECK(rasterize_triangles_v2(imgD1, W, H, CH, &t, 1, 1, TRI_AA_ANALYTICAL, TRI_PRECISION_FIXED8) == TRI_OK,
              "precision: det 1t ok");
        CHECK(rasterize_triangles_v2(imgD8, W, H, CH, &t, 1, 8, TRI_AA_ANALYTICAL, TRI_PRECISION_FIXED8) == TRI_OK,
              "precision: det 8t ok");
        CHECK(buf_eq(imgD1, imgD8, W * H * CH),
              "precision: fixed8 1 thread == 8 threads (bit-identical)");
    }

    if (g_failures == 0) {
        printf("[PASS] all tests passed\n");
        return 0;
    }
    printf("[FAIL] %d test(s) failed\n", g_failures);
    return 1;
}
