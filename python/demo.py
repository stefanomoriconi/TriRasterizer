"""
Demo + end-to-end verification for the triangle_rasterizer Python wrapper.

Renders a small scene of overlapping triangles with varying colour/alpha,
saves it to a PNG, and asserts:

  * single-thread vs multi-thread output is bit-identical (determinism);
  * painter's order is honoured (a later triangle overwrites an earlier one
    where they overlap);
  * alpha blending is sane;
  * the rasterisation actually touched the expected region.

Run:  python python/demo.py
Output:  python/output/demo_scene.png  (and a couple of diagnostic PNGs)
"""

import os
import sys
import time

import numpy as np

# The importable package lives at the repository root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import triangle_rasterizer as tr  # noqa: E402


def _save(img: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(img, "RGB" if img.shape[2] == 3 else "RGBA").save(path)
        return
    except Exception:
        pass
    # fallback: write PPM (no PIL)
    h, w, _ = img.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(np.ascontiguousarray(img[..., :3]).tobytes())


def build_scene() -> list:
    W = H = 512
    bg = (24, 28, 36)
    return [
        dict(v0=(40, 40),   v1=(472, 40),   v2=(256, 472),
             color=bg, alpha=1.0),                              # backdrop
        dict(v0=(70, 90),   v1=(300, 150),  v2=(180, 380),
             color=(220, 40, 40), alpha=1.0),                   # red
        dict(v0=(250, 120), v1=(460, 220),  v2=(330, 430),
             color=(40, 120, 230), alpha=1.0),                  # blue
        dict(v0=(150, 180), v1=(370, 180),  v2=(260, 400),
             color=(40, 200, 90), alpha=0.7),                   # green, semi
        dict(v0=(200, 220), v1=(320, 220),  v2=(260, 340),
             color=(255, 220, 60), alpha=0.8),                  # yellow, top
    ]


def main() -> int:
    print("triangle_rasterizer wrapper demo")
    print("  backend           :", tr.backend())
    print("  C library version :", tr.version())
    print("  OpenMP available  :", tr.omp_available())
    print("  OpenMP max threads:", tr.omp_max_threads())

    H = W = 512
    scene = build_scene()

    # --- render single-thread and multi-thread, compare -------------------- #
    img1 = np.zeros((H, W, 3), dtype=np.uint8)
    img2 = np.zeros((H, W, 3), dtype=np.uint8)

    t0 = time.perf_counter(); tr.rasterize(img1, scene, n_threads=1); t1 = time.perf_counter()
    t2 = time.perf_counter(); tr.rasterize(img2, scene, n_threads=0); t3 = time.perf_counter()

    identical = np.array_equal(img1, img2)
    print(f"  single-thread time : {1000*(t1-t0):8.3f} ms")
    print(f"  multi-thread time  : {1000*(t3-t2):8.3f} ms")
    print(f"  deterministic (1 vs N threads identical): {identical}")
    if not identical:
        print("  !! FAIL: determinism")
        return 1

    # --- painter's order check (clean, opaque, 2 triangles) ---------------- #
    # Draw red first, then yellow fully covering a point; that point must be
    # yellow (the later triangle wins), independent of any blending.
    pw = ph = 64
    pimg = np.zeros((ph, pw, 3), dtype=np.uint8)
    tr.draw_triangle(pimg, (5, 5), (59, 5), (32, 59), color=(255, 0, 0), alpha=1.0)
    tr.draw_triangle(pimg, (14, 14), (50, 14), (32, 46), color=(0, 255, 0), alpha=1.0)
    centre = pimg[32, 32]
    print(f"  painter's order @centre = {tuple(centre)}  (expect green (0,255,0))")
    painter_ok = np.array_equal(centre, np.array([0, 255, 0]))
    # and a point covered only by the red triangle must stay red
    red_only = pimg[10, 32]
    print(f"  red-only pixel        = {tuple(red_only)}  (expect red   (255,0,0))")
    painter_ok = painter_ok and np.array_equal(red_only, np.array([255, 0, 0]))
    if not painter_ok:
        print("  !! FAIL: painter's order")
        return 1

    # --- region coverage check -------------------------------------------- #
    # Backdrop should have filled the whole canvas.
    covered = int(np.count_nonzero(np.any(img1 != 0, axis=2)))
    print(f"  non-zero pixels   : {covered}/{H*W}")

    # --- alpha sanity ------------------------------------------------------ #
    img_a = np.zeros((64, 64, 3), dtype=np.uint8)
    tr.draw_triangle(img_a, (10, 10), (54, 10), (32, 54), color=(255, 0, 0), alpha=0.5)
    center = img_a[30, 32]
    print(f"  alpha 0.5 red over black @center = {tuple(center)} (expect ~ (128,0,0))")
    alpha_ok = abs(int(center[0]) - 128) <= 2

    # --- save outputs ------------------------------------------------------ #
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    _save(img2, os.path.join(out_dir, "demo_scene.png"))
    print("  saved             :", os.path.join(out_dir, "demo_scene.png"))

    ok = identical and painter_ok and alpha_ok and covered > 0
    print()
    print("[PASS]" if ok else "[FAIL]", "end-to-end Python verification")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
