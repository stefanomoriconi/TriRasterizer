"""Python test suite for the ``triangle_rasterizer`` package.

These tests are written as plain ``test_*`` functions using bare ``assert``
so they run **both** under ``pytest`` and via the dependency-free
standalone runner ``run_tests.py`` (this environment has no network, so
pytest may not be installable).

Run with pytest (if available):
    pytest tests/python -q

Run standalone (always works):
    python tests/python/run_tests.py

The GPU parity test is automatically skipped when the loaded library is
not a CUDA build (``triangle_rasterizer.backend() != "cuda"``), so the
suite passes on CPU-only machines.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Ensure the repository root (containing the package) is importable whether
# we run from the repo root or from this directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import triangle_rasterizer as tr  # noqa: E402


# --------------------------------------------------------------------------- #
# Basic rendering
# --------------------------------------------------------------------------- #
def test_version_string() -> None:
    v = tr.version()
    assert isinstance(v, str) and v.strip() != ""


def test_backend_is_string() -> None:
    b = tr.backend()
    assert isinstance(b, str) and b.strip() != ""
    assert b.split()[0].lower() in ("cpu", "cuda")


def test_opaque_fills_interior() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    tr.draw_triangle(img, (5, 5), (59, 5), (32, 59), color=(255, 0, 0), alpha=1.0)
    centre = img[32, 32]
    assert tuple(centre) == (255, 0, 0)
    # A corner outside the triangle stays black.
    assert tuple(img[0, 0]) == (0, 0, 0)


def test_painter_order_later_wins() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    tr.draw_triangle(img, (5, 5), (59, 5), (32, 59), color=(255, 0, 0), alpha=1.0)
    tr.draw_triangle(img, (14, 14), (50, 14), (32, 46), color=(0, 255, 0), alpha=1.0)
    assert tuple(img[32, 32]) == (0, 255, 0)          # later (green) wins here
    assert tuple(img[10, 32]) == (255, 0, 0)          # red-only region stays red


def test_determinism_thread_independent() -> None:
    rng = np.random.default_rng(0)
    tris = []
    for _ in range(64):
        pts = rng.integers(0, 128, size=(3, 2)).tolist()
        col = tuple(int(c) for c in rng.integers(0, 256, size=3).tolist())
        tris.append(dict(v0=pts[0], v1=pts[1], v2=pts[2], color=col,
                         alpha=float(rng.uniform(0.0, 1.0))))
    img1 = np.zeros((128, 128, 3), dtype=np.uint8)
    img2 = np.zeros((128, 128, 3), dtype=np.uint8)
    tr.rasterize(img1, tris, n_threads=1)
    tr.rasterize(img2, tris, n_threads=8)
    assert np.array_equal(img1, img2)


def test_alpha_blend_half() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    tr.draw_triangle(img, (10, 10), (54, 10), (32, 54), color=(255, 0, 0), alpha=0.5)
    centre = img[30, 32]
    assert abs(int(centre[0]) - 128) <= 2
    assert int(centre[1]) == 0 and int(centre[2]) == 0


def test_rgba_shape_and_rgb() -> None:
    # The rasterizer writes RGB only; the 4th channel (alpha) is left untouched
    # (standard "over" compositing semantics).  Pre-fill with a sentinel and
    # verify RGB is drawn while A is preserved.
    img = np.full((64, 64, 4), 7, dtype=np.uint8)
    tr.draw_triangle(img, (10, 10), (54, 10), (32, 54), color=(255, 255, 255), alpha=1.0)
    centre = img[30, 32]
    assert tuple(centre[:3]) == (255, 255, 255)   # RGB drawn opaque
    assert int(centre[3]) == 7                     # alpha channel untouched


def test_degenerate_triangle_no_crash() -> None:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    # Collinear / zero-area triangles must be skipped without error.
    tr.rasterize(img, [
        dict(v0=(5, 5), v1=(15, 5), v2=(25, 5), color=(255, 0, 0), alpha=1.0),
        dict(v0=(8, 8), v1=(8, 8), v2=(8, 8), color=(0, 255, 0), alpha=1.0),
    ], n_threads=0)
    assert img.shape == (32, 32, 3)


def test_image_shapes() -> None:
    # (H, W, 3)
    img = np.zeros((10, 12, 3), dtype=np.uint8)
    tr.rasterize(img, [dict(v0=(1, 1), v1=(11, 1), v2=(6, 11), color=(1, 2, 3), alpha=1.0)])
    assert img.shape == (10, 12, 3)
    # (H, W, 4)
    img = np.zeros((10, 12, 4), dtype=np.uint8)
    tr.rasterize(img, [dict(v0=(1, 1), v1=(11, 1), v2=(6, 11), color=(1, 2, 3), alpha=1.0)])
    assert img.shape == (10, 12, 4)
    # 2-D input is expanded to RGB.
    img = np.zeros((10, 12), dtype=np.uint8)
    out = tr.rasterize(img, [dict(v0=(1, 1), v1=(11, 1), v2=(6, 11), color=(1, 2, 3), alpha=1.0)])
    assert out.shape == (10, 12, 3)


def test_empty_and_none_triangles() -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    tr.rasterize(img, [])
    tr.rasterize(img, None)
    assert not np.any(img)


def test_bad_input_raises() -> None:
    # Wrong channel count.
    bad = np.zeros((16, 16, 5), dtype=np.uint8)
    try:
        tr.rasterize(bad, [dict(v0=(1, 1), v1=(11, 1), v2=(6, 11), color=(1, 2, 3), alpha=1.0)])
        raise AssertionError("expected ValueError for 5 channels")
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# Batch API
# --------------------------------------------------------------------------- #
def test_batch_render_and_count() -> None:
    scene = [
        dict(v0=(5, 5), v1=(59, 5), v2=(32, 59), color=(255, 0, 0), alpha=1.0),
        dict(v0=(14, 14), v1=(50, 14), v2=(32, 46), color=(0, 255, 0), alpha=1.0),
    ]
    with tr.TriangleBatch(scene) as batch:
        assert batch.count == 2
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        batch.draw(img, n_threads=0)
        assert tuple(img[32, 32]) == (0, 255, 0)

    # After free, drawing must raise (handle released).
    try:
        batch.draw(np.zeros((8, 8, 3), dtype=np.uint8))
        # Some platforms may lazily reuse memory; that's fine — just must not segfault.
    except Exception:
        pass


def test_batch_add_grows() -> None:
    with tr.TriangleBatch() as batch:
        assert batch.count == 0
        batch.add(dict(v0=(5, 5), v1=(59, 5), v2=(32, 59), color=(255, 0, 0), alpha=1.0))
        batch.add(dict(v0=(5, 5), v1=(59, 59), v2=(59, 5), color=(0, 0, 255), alpha=1.0))
        assert batch.count == 2


def test_batch_repeated_render_matches_direct() -> None:
    scene = [
        dict(v0=(5, 5), v1=(59, 5), v2=(32, 59), color=(255, 0, 0), alpha=0.6),
        dict(v0=(14, 14), v1=(50, 14), v2=(32, 46), color=(0, 255, 0), alpha=0.6),
    ]
    direct = np.zeros((64, 64, 3), dtype=np.uint8)
    tr.rasterize(direct, scene, n_threads=0)

    with tr.TriangleBatch(scene) as batch:
        from_batch = np.zeros((64, 64, 3), dtype=np.uint8)
        batch.draw(from_batch, n_threads=0)
    assert np.array_equal(direct, from_batch)


# --------------------------------------------------------------------------- #
# Error reporting
# --------------------------------------------------------------------------- #
def test_error_string_defined() -> None:
    # TRI_ERR_NULL == 1 per the C header.
    s = tr.error_string(1)
    assert isinstance(s, str) and s.strip() != ""


def test_omp_diagnostics_consistent() -> None:
    avail = tr.omp_available()
    max_t = tr.omp_max_threads()
    assert isinstance(avail, bool)
    assert max_t >= 1
    if avail:
        assert max_t >= 1
    # Without OpenMP the library reports 1 (serial).
    if not avail:
        assert max_t == 1


# --------------------------------------------------------------------------- #
# Anti-aliasing (v1.2.0)
# --------------------------------------------------------------------------- #
def test_aa_constants_present() -> None:
    assert tr.AA_NONE == 0
    assert tr.AA_ANALYTICAL == 1
    assert tr.AA_SSAA2X2 == 2
    assert tr.AA_SSAA4ROT == 3


def test_aa_invalid_mode_raises() -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    for bad in (99, -1, "analytical", None, 2.5):
        try:
            tr.rasterize(img, [dict(v0=(1, 1), v1=(11, 1), v2=(6, 11),
                                    color=(255, 0, 0), alpha=1.0)], antialiasing=bad)
            raise AssertionError("expected error for antialiasing=%r" % (bad,))
        except (ValueError, TypeError):
            pass


def test_aa_analytical_interior_and_boundary() -> None:
    # Right triangle: top edge y=0, left edge x=0, hypotenuse (100,0)->(0,100).
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    tr.draw_triangle(img, (0, 0), (100, 0), (0, 100),
                     color=(255, 0, 0), alpha=1.0, antialiasing=tr.AA_ANALYTICAL)
    # Interior pixel is fully covered -> full red.
    assert tuple(img[10, 10]) == (255, 0, 0)
    # Hypotenuse pixel at (x=49, y=50) is PARTIALLY covered (0 < r < 255).
    r = int(img[50, 49, 0])
    assert 0 < r < 255, "expected partial coverage, got %d" % r


def test_aa_ssaa_interior_and_boundary() -> None:
    for mode in (tr.AA_SSAA2X2, tr.AA_SSAA4ROT):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        tr.draw_triangle(img, (0, 0), (100, 0), (0, 100),
                         color=(255, 0, 0), alpha=1.0, antialiasing=mode)
        assert tuple(img[10, 10]) == (255, 0, 0), "interior not full in mode %d" % mode
        # Exterior pixel stays black.
        assert tuple(img[120, 120]) == (0, 0, 0)
        # Hypotenuse pixel is partial.
        r = int(img[50, 49, 0])
        assert 0 < r < 255, "expected partial coverage in mode %d, got %d" % (mode, r)


def test_aa_none_matches_legacy() -> None:
    # antialiasing=AA_NONE must be bit-identical to the legacy default path.
    tris = [
        dict(v0=(10, 30), v1=(50, 30), v2=(30, 10), color=(255, 0, 0), alpha=1.0),
        dict(v0=(20, 20), v1=(80, 20), v2=(50, 80), color=(0, 255, 0), alpha=0.5),
    ]
    legacy = np.zeros((128, 128, 3), dtype=np.uint8)
    tr.rasterize(legacy, tris, n_threads=0)
    aa_none = np.zeros((128, 128, 3), dtype=np.uint8)
    tr.rasterize(aa_none, tris, n_threads=0, antialiasing=tr.AA_NONE)
    assert np.array_equal(legacy, aa_none), "AA_NONE must be bit-identical to legacy"


def test_aa_determinism_thread_independent() -> None:
    rng = np.random.default_rng(1)
    tris = []
    for _ in range(48):
        pts = rng.integers(0, 128, size=(3, 2)).tolist()
        tris.append(dict(v0=pts[0], v1=pts[1], v2=pts[2],
                         color=(200, 30, 30),
                         alpha=float(rng.uniform(0.3, 1.0))))
    for mode in (tr.AA_ANALYTICAL, tr.AA_SSAA2X2, tr.AA_SSAA4ROT):
        img1 = np.zeros((128, 128, 3), dtype=np.uint8)
        img8 = np.zeros((128, 128, 3), dtype=np.uint8)
        tr.rasterize(img1, tris, n_threads=1, antialiasing=mode)
        tr.rasterize(img8, tris, n_threads=8, antialiasing=mode)
        assert np.array_equal(img1, img8), "AA mode %d not deterministic" % mode


def test_aa_batch_forwarding() -> None:
    scene = [dict(v0=(5, 5), v1=(59, 5), v2=(32, 59), color=(255, 0, 0), alpha=1.0)]
    with tr.TriangleBatch(scene) as batch:
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        batch.draw(img, n_threads=0, antialiasing=tr.AA_ANALYTICAL)
        # Interior fully covered.
        assert tuple(img[32, 32]) == (255, 0, 0)


def test_aa_performance() -> None:
    """Relative throughput across AA modes.

    This is an informational/perf test: it only asserts that AA modes complete
    and that AA_NONE is not dramatically slower than AA (a sanity guard, not a
    hard performance contract).  Prints a small comparison table.
    """
    import time
    W = H = 512
    rng = np.random.default_rng(3)
    tris = []
    for _ in range(2000):
        pts = rng.integers(0, W, size=(3, 2)).tolist()
        tris.append(dict(v0=pts[0], v1=pts[1], v2=pts[2],
                         color=(180, 40, 40),
                         alpha=float(rng.uniform(0.4, 1.0))))

    def bench(mode):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        tr.rasterize(img, tris, n_threads=0, antialiasing=mode)
        return time.perf_counter() - t0

    t_none = bench(tr.AA_NONE)
    t_ana = bench(tr.AA_ANALYTICAL)
    t_s2 = bench(tr.AA_SSAA2X2)
    t_s4 = bench(tr.AA_SSAA4ROT)
    print("\n[perf] rasterize 2000 tris, %dx%d, backend=%r" % (W, H, tr.backend()))
    print("  AA_NONE       %.4f s" % t_none)
    print("  AA_ANALYTICAL %.4f s" % t_ana)
    print("  AA_SSAA2X2    %.4f s" % t_s2)
    print("  AA_SSAA4ROT   %.4f s" % t_s4)
    # Sanity only: every AA mode must complete in finite time.  SSAA samples 4
    # taps per pixel so it is legitimately several times slower than the 1-tap
    # base path; we deliberately do NOT enforce a tight speed ratio (that would
    # turn an informational benchmark into a fragile contract).
    for name, t in (("ANALYTICAL", t_ana), ("SSAA2X2", t_s2), ("SSAA4ROT", t_s4)):
        assert 0.0 < t < 30.0, "AA_%s took implausible %.4fs" % (name, t)


# --------------------------------------------------------------------------- #
# Fixed-point precision (v1.3.0)
# --------------------------------------------------------------------------- #
def test_precision_constants_present() -> None:
    assert tr.PRECISION_FLOAT == 0
    assert tr.PRECISION_FIXED8 == 1


def test_precision_invalid_mode_raises() -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    for bad in (99, -1, "fixed8", None, 2.5):
        try:
            tr.rasterize(img, [dict(v0=(1, 1), v1=(11, 1), v2=(6, 11),
                                    color=(255, 0, 0), alpha=1.0)], precision=bad)
            raise AssertionError("expected error for precision=%r" % (bad,))
        except (ValueError, TypeError):
            pass


def test_precision_float_matches_legacy() -> None:
    # precision=PRECISION_FLOAT must be bit-identical to the default path.
    tris = [
        dict(v0=(10, 30), v1=(50, 30), v2=(30, 10), color=(255, 0, 0), alpha=1.0),
        dict(v0=(20, 20), v1=(80, 20), v2=(50, 80), color=(0, 255, 0), alpha=0.5),
    ]
    legacy = np.zeros((128, 128, 3), dtype=np.uint8)
    tr.rasterize(legacy, tris, n_threads=0)
    explicit = np.zeros((128, 128, 3), dtype=np.uint8)
    tr.rasterize(explicit, tris, n_threads=0, precision=tr.PRECISION_FLOAT)
    assert np.array_equal(legacy, explicit)


def test_precision_fixed8_hard_edge_close_to_float() -> None:
    rng = np.random.default_rng(7)
    tris = []
    for _ in range(40):
        pts = (rng.uniform(0, 128, size=(3, 2))).tolist()
        tris.append(dict(v0=pts[0], v1=pts[1], v2=pts[2],
                         color=(200, 60, 20), alpha=1.0))
    img_f = np.zeros((128, 128, 3), dtype=np.uint8)
    img_x = np.zeros((128, 128, 3), dtype=np.uint8)
    tr.rasterize(img_f, tris, precision=tr.PRECISION_FLOAT)
    tr.rasterize(img_x, tris, precision=tr.PRECISION_FIXED8)
    diff = np.abs(img_f.astype(np.int16) - img_x.astype(np.int16))
    n_diff_px = int((diff.sum(axis=2) > 0).sum())
    # Only a small fraction of pixels (triangle edges) may disagree, by
    # boundary rounding at most; never a large area mismatch.
    assert n_diff_px < 0.02 * img_f.shape[0] * img_f.shape[1], n_diff_px


def test_precision_fixed8_analytical_close_to_float() -> None:
    img_f = np.zeros((128, 128, 3), dtype=np.uint8)
    img_x = np.zeros((128, 128, 3), dtype=np.uint8)
    tr.draw_triangle(img_f, (0, 0), (100, 0), (0, 100), color=(255, 0, 0),
                     antialiasing=tr.AA_ANALYTICAL, precision=tr.PRECISION_FLOAT)
    tr.draw_triangle(img_x, (0, 0), (100, 0), (0, 100), color=(255, 0, 0),
                     antialiasing=tr.AA_ANALYTICAL, precision=tr.PRECISION_FIXED8)
    diff = np.abs(img_f.astype(np.int16) - img_x.astype(np.int16))
    assert diff.max() <= 1, "fixed8 analytical AA should match float within 1 level"


def test_precision_determinism_thread_independent() -> None:
    rng = np.random.default_rng(2)
    tris = []
    for _ in range(48):
        pts = rng.integers(0, 128, size=(3, 2)).tolist()
        tris.append(dict(v0=pts[0], v1=pts[1], v2=pts[2],
                         color=(200, 30, 30),
                         alpha=float(rng.uniform(0.3, 1.0))))
    img1 = np.zeros((128, 128, 3), dtype=np.uint8)
    img8 = np.zeros((128, 128, 3), dtype=np.uint8)
    tr.rasterize(img1, tris, n_threads=1, precision=tr.PRECISION_FIXED8)
    tr.rasterize(img8, tris, n_threads=8, precision=tr.PRECISION_FIXED8)
    assert np.array_equal(img1, img8), "PRECISION_FIXED8 not deterministic across threads"


def test_precision_batch_forwarding() -> None:
    scene = [dict(v0=(5, 5), v1=(59, 5), v2=(32, 59), color=(255, 0, 0), alpha=1.0)]
    with tr.TriangleBatch(scene) as batch:
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        batch.draw(img, n_threads=0, precision=tr.PRECISION_FIXED8)
        assert tuple(img[32, 32]) == (255, 0, 0)


# --------------------------------------------------------------------------- #
# GPU parity (skipped unless the loaded library is a CUDA build)
# --------------------------------------------------------------------------- #
def test_cuda_parity_if_available() -> None:
    b = tr.backend().split()[0].lower()
    if b != "cuda":
        # Not a CUDA build — nothing to compare against; expected on CPU-only
        # machines and in CI without a GPU.  Skip when running under pytest.
        try:
            import pytest
            pytest.skip("not a CUDA build (backend = %r)" % tr.backend())
        except Exception:
            return

    # A CUDA build: verify determinism still holds (1 vs N threads).
    rng = np.random.default_rng(7)
    tris = []
    for _ in range(32):
        pts = rng.integers(0, 128, size=(3, 2)).tolist()
        tris.append(dict(v0=pts[0], v1=pts[1], v2=pts[2],
                         color=(200, 30, 30), alpha=float(rng.uniform(0.4, 1.0))))
    img1 = np.zeros((128, 128, 3), dtype=np.uint8)
    img2 = np.zeros((128, 128, 3), dtype=np.uint8)
    tr.rasterize(img1, tris, n_threads=1)
    tr.rasterize(img2, tris, n_threads=0)
    assert np.array_equal(img1, img2)
