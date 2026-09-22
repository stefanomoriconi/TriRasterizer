"""Fixed-point vs floating-point precision comparison (v1.3.0).

Renders the SAME scene with both scanline precision modes
(``PRECISION_FLOAT`` legacy double maths and ``PRECISION_FIXED8`` Q24.8
OpenCV-style fixed-point maths), for every anti-aliasing mode, and reports:

  1. Accuracy   — a per-mode diff image + max/mean absolute pixel difference
                  between PRECISION_FLOAT and PRECISION_FIXED8.
  2. Performance — a side-by-side timing table across the full
                  {precision} x {antialiasing} matrix on a larger scene.

Note: FIXED8 is expected to be marginally faster than FLOAT for the 1-tap
paths (``AA_NONE``, ``AA_ANALYTICAL``), where per-row integer arithmetic
replaces per-row double division.  It can be *slower* for the 4-tap SSAA
paths on some platforms, because quantising each sub-pixel tap with
``lround()`` costs more than the plain double add the float path uses —
that trade-off is reported here, not hidden.

Run:  python examples/precision_compare.py
Saves: examples/output/precision_compare.png
"""

from __future__ import annotations

import os
import time

import numpy as np

import common  # noqa: F401  (sets up sys.path + helpers)
from common import ensure_output_dir, report, save_image

import triangle_rasterizer as tr

SCENE = [
    dict(v0=(40, 60),   v1=(460, 60),   v2=(250, 300),
         color=(235, 238, 242), alpha=1.0),
    dict(v0=(90, 120),  v1=(360, 120),  v2=(225, 360),
         color=(220, 60, 40),  alpha=1.0),
    dict(v0=(200, 150), v1=(420, 240),  v2=(300, 400),
         color=(40, 110, 230), alpha=1.0),
    dict(v0=(150, 200), v1=(340, 210),  v2=(250, 390),
         color=(60, 200, 90),  alpha=0.75),
]

AA_MODES = [
    (tr.AA_NONE,       "NONE"),
    (tr.AA_ANALYTICAL, "ANALYTICAL"),
    (tr.AA_SSAA2X2,    "SSAA2X2"),
    (tr.AA_SSAA4ROT,   "SSAA4ROT"),
]

PRECISIONS = [
    (tr.PRECISION_FLOAT,  "FLOAT"),
    (tr.PRECISION_FIXED8, "FIXED8"),
]


def _render(precision: int, aa: int, w: int, h: int) -> "np.ndarray":
    img = np.full((h, w, 3), 24, dtype=np.uint8)
    tr.rasterize(img, SCENE, n_threads=0, antialiasing=aa, precision=precision)
    return img


def _accuracy_report(w: int = 420, h: int = 420) -> "np.ndarray":
    """Build a composite image: for each AA mode, FLOAT | FIXED8 | 8x diff."""
    gap = 8
    label_h = 22
    n = len(AA_MODES)
    row_h = h + label_h + gap
    board = np.full((row_h * n + gap, 3 * w + 4 * gap, 3), 32, dtype=np.uint8)

    print("accuracy: PRECISION_FLOAT vs PRECISION_FIXED8 (per AA mode)")
    print(f"{'mode':<12} {'max|diff|':>10} {'mean|diff|':>12} {'%px differing':>15}")
    for row, (aa, aa_name) in enumerate(AA_MODES):
        img_f = _render(tr.PRECISION_FLOAT, aa, w, h)
        img_x = _render(tr.PRECISION_FIXED8, aa, w, h)
        diff = np.abs(img_f.astype(np.int16) - img_x.astype(np.int16))
        max_d = int(diff.max())
        mean_d = float(diff.mean())
        pct_differing = 100.0 * float((diff.sum(axis=2) > 0).mean())
        print(f"{aa_name:<12} {max_d:>10d} {mean_d:>12.4f} {pct_differing:>14.2f}%")

        diff_vis = np.clip(diff.astype(np.int32) * 8, 0, 255).astype(np.uint8)  # 8x boost
        y0 = gap + row * row_h
        board[y0 + label_h: y0 + label_h + h, gap: gap + w] = img_f
        board[y0 + label_h: y0 + label_h + h, 2 * gap + w: 2 * gap + 2 * w] = img_x
        board[y0 + label_h: y0 + label_h + h, 3 * gap + 2 * w: 3 * gap + 3 * w] = diff_vis
    print()
    return board


def _perf_report(w: int = 512, h: int = 512, n_tris: int = 2000) -> None:
    rng = np.random.default_rng(3)
    tris = []
    for _ in range(n_tris):
        pts = rng.integers(0, w, size=(3, 2)).tolist()
        tris.append(dict(v0=pts[0], v1=pts[1], v2=pts[2],
                         color=(180, 40, 40), alpha=float(rng.uniform(0.4, 1.0))))

    print(f"performance: {n_tris} tris, {w}x{h}, backend={tr.backend()!r}")
    print(f"{'AA mode':<12} {'FLOAT (s)':>12} {'FIXED8 (s)':>12} {'fixed8/float':>14}")
    for aa, aa_name in AA_MODES:
        times = {}
        for precision, prec_name in PRECISIONS:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            tr.rasterize(img, tris, n_threads=0, antialiasing=aa, precision=precision)
            times[prec_name] = time.perf_counter() - t0
        ratio = times["FIXED8"] / times["FLOAT"] if times["FLOAT"] > 0 else float("nan")
        print(f"{aa_name:<12} {times['FLOAT']:>12.4f} {times['FIXED8']:>12.4f} {ratio:>13.2f}x")
    print()


def main() -> int:
    report()
    board = _accuracy_report()
    _perf_report()

    out = os.path.join(ensure_output_dir(), "precision_compare.png")
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(np.ascontiguousarray(board), "RGB")
        d = ImageDraw.Draw(img)
        font = None
        try:
            font = ImageFont.load_default()
        except Exception:
            pass
        w = 420
        gap = 8
        for row, (_aa, name) in enumerate(AA_MODES):
            y0 = gap + row * (420 + 22 + gap)
            d.text((gap + 4, y0 + 2), f"{name}: FLOAT", fill=(235, 235, 235), font=font)
            d.text((2 * gap + w + 4, y0 + 2), f"{name}: FIXED8", fill=(235, 235, 235), font=font)
            d.text((3 * gap + 2 * w + 4, y0 + 2), f"{name}: |diff|x8", fill=(235, 235, 235), font=font)
        img.save(out)
    except Exception:
        save_image(board, out)
    print(f"comparison -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
