"""Basic rendering example.

Draws a few opaque and semi-transparent triangles into an RGB image and
saves the result to ``examples/output/basic_render.png``.

Run:  python examples/basic_render.py
"""

import os

import numpy as np

import common  # noqa: F401  (sets up sys.path + helpers)
from common import ensure_output_dir, report, save_image

import triangle_rasterizer as tr


def main() -> int:
    report()

    W = H = 512
    img = np.zeros((H, W, 3), dtype=np.uint8)

    triangles = [
        # (v0, v1, v2, color, alpha)  — colours are 0..255, alpha is 0..1
        dict(v0=(40, 40),   v1=(472, 40),   v2=(256, 472),
             color=(24, 28, 36),    alpha=1.0),   # dark backdrop
        dict(v0=(70, 90),   v1=(300, 150),  v2=(180, 380),
             color=(220, 40, 40),   alpha=1.0),   # red
        dict(v0=(250, 120), v1=(460, 220),  v2=(330, 430),
             color=(40, 120, 230),  alpha=1.0),   # blue
        dict(v0=(150, 180), v1=(370, 180),  v2=(260, 400),
             color=(40, 200, 90),   alpha=0.7),   # green, semi-transparent
        dict(v0=(200, 220), v1=(320, 220),  v2=(260, 340),
             color=(255, 220, 60),  alpha=0.8),   # yellow, on top
    ]

    tr.rasterize(img, triangles, n_threads=0)  # 0 -> let the library decide

    out = os.path.join(ensure_output_dir(), "basic_render.png")
    save_image(img, out)
    covered = int(np.count_nonzero(np.any(img != 0, axis=2)))
    print(f"rendered {len(triangles)} triangles -> {out}")
    print(f"non-zero pixels: {covered}/{H * W}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
