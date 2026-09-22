"""Anti-aliasing comparison (v1.2.0).

Renders the SAME scene with each anti-aliasing mode and lays the results out
side-by-side in a single comparison image, plus a magnified crop of a slanted
edge so the difference in edge quality is easy to see.

Modes compared (``triangle_rasterizer`` constants):
    AA_NONE        (0)  legacy hard edges  (no anti-aliasing)
    AA_ANALYTICAL  (1)  exact per-pixel geometric coverage (1 tap)
    AA_SSAA2X2     (2)  2x2 supersampling (4 taps)
    AA_SSAA4ROT    (3)  4 rotated supersamples (4 taps)

Run:  python examples/antialiasing_compare.py
Saves: examples/output/antialiasing_compare.png
"""

from __future__ import annotations

import os

import numpy as np

import common  # noqa: F401  (sets up sys.path + helpers)
from common import ensure_output_dir, report, save_image

import triangle_rasterizer as tr

# A fixed scene chosen to contain several slanted edges so anti-aliasing is
# visible.  Colours are 0..255, alpha is 0..1.
SCENE = [
    dict(v0=(40, 60),   v1=(460, 60),   v2=(250, 300),
         color=(235, 238, 242),  alpha=1.0),   # light face
    dict(v0=(90, 120),  v1=(360, 120),  v2=(225, 360),
         color=(220, 60, 40),   alpha=1.0),   # red
    dict(v0=(200, 150), v1=(420, 240),  v2=(300, 400),
         color=(40, 110, 230),  alpha=1.0),   # blue
    dict(v0=(150, 200), v1=(340, 210),  v2=(250, 390),
         color=(60, 200, 90),   alpha=0.75),  # green, semi-transparent
]

MODES = [
    (tr.AA_NONE,       "NONE (legacy)"),
    (tr.AA_ANALYTICAL, "ANALYTICAL"),
    (tr.AA_SSAA2X2,    "SSAA 2x2"),
    (tr.AA_SSAA4ROT,   "SSAA 4rot"),
]


def _render(mode: int, w: int, h: int) -> "np.ndarray":
    img = np.full((h, w, 3), 24, dtype=np.uint8)  # dark backdrop
    tr.rasterize(img, SCENE, n_threads=0, antialiasing=mode)
    return img


def main() -> int:
    report()

    # --- Full-size side-by-side panels (with a gap + title band). ----------- #
    W = H = 420
    gap = 8
    label_h = 26
    n = len(MODES)
    board_w = n * W + (n + 1) * gap
    board_h = label_h + H + gap * 2

    board = np.full((board_h, board_w, 3), 32, dtype=np.uint8)
    for i, (mode, _name) in enumerate(MODES):
        x0 = gap + i * (W + gap)
        panel = _render(mode, W, H)
        board[label_h + gap : label_h + gap + H, x0 : x0 + W] = panel

    # --- Magnified edge crop (nearest-neighbour upscale of a slanted edge). -- #
    # Pick a small window around a slanted edge shared by the scene, then
    # upscale it ~6x so individual pixel steps are visible.
    cw, ch = 40, 40
    scale = 6
    crop_board_w = n * (cw * scale) + (n + 1) * gap
    crop_board_h = label_h + ch * scale + gap * 2
    crop_board = np.full((crop_board_h, crop_board_w, 3), 32, dtype=np.uint8)
    for i, (mode, _name) in enumerate(MODES):
        full = _render(mode, W, H)
        # A slanted interior edge lives around (x=210, y=200) in the scene.
        cx, cy = 190, 180
        crop = full[cy : cy + ch, cx : cx + cw]
        # Nearest-neighbour upscale (repeat each pixel into a block).
        up = np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1)
        x0 = gap + i * (cw * scale + gap)
        crop_board[label_h + gap : label_h + gap + up.shape[0],
                   x0 : x0 + up.shape[1]] = up

    # --- Compose final image (full row on top, magnified crop below). -------- #
    final_h = board_h + crop_board_h + gap
    final = np.empty((final_h, max(board_w, crop_board_w), 3), dtype=np.uint8)
    final[:] = 32
    final[:board_h, :board_w] = board
    final[board_h + gap : board_h + gap + crop_board_h, :crop_board_w] = crop_board

    # --- Text labels (PIL only; harmless to skip). -------------------------- #
    out = os.path.join(ensure_output_dir(), "antialiasing_compare.png")
    try:
        from PIL import Image, ImageDraw
        img = Image.fromarray(np.ascontiguousarray(final), "RGB")
        d = ImageDraw.Draw(img)
        font = None
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
        except Exception:
            pass
        for i, (_mode, name) in enumerate(MODES):
            x0 = gap + i * (W + gap)
            d.text((x0 + 4, 6), name, fill=(235, 235, 235), font=font)
            cxx = gap + i * (cw * scale + gap)
            d.text((cxx + 4, board_h + gap + 6), name, fill=(235, 235, 235), font=font)
        img.save(out)
        print(f"comparison (labelled) -> {out}")
    except Exception:
        save_image(final, out)
        print(f"comparison (no labels) -> {out}")

    # --- Per-mode edge statistics (quantitative, no PIL required). ---------- #
    print("\nedge-softness metrics over the shared slanted region (lower std = harder):")
    cx, cy = 150, 150
    sw = 90
    for mode, name in MODES:
        full = _render(mode, W, H)
        region = full[cy : cy + sw, cx : cx + sw].astype(np.float64)
        # Count distinct grey levels in the luminance channel -> proxy for how
        # many intermediate coverage values the mode produces on the edges.
        lum = 0.299 * region[..., 0] + 0.587 * region[..., 1] + 0.114 * region[..., 2]
        n_levels = int(np.unique(np.round(lum)).size)
        # Fraction of non-background, non-full pixels (anti-aliased edge pixels).
        bg = 24.0
        edge_frac = float(np.mean((np.abs(lum - bg) > 4) & (lum < 200)))
        print(f"  {name:<12} distinct_levels={n_levels:3d}  edge_pixel_frac={edge_frac:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
