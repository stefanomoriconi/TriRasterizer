"""Shared helpers for the examples/ scripts.

These make each example runnable straight from a checkout (``python
examples/basic_render.py``) without requiring the package to be installed.
"""

from __future__ import annotations

import os
import sys

# Ensure the repository root (which contains the importable package) is
# importable when running scripts directly from a checkout.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def ensure_output_dir(*parts: str) -> str:
    out = os.path.join(_ROOT, "examples", "output", *parts)
    os.makedirs(out, exist_ok=True)
    return out


def save_image(img, path: str) -> str:
    """Save an RGB(A) uint8 image to ``path``.  Uses PIL when available,
    otherwise falls back to a plain PPM write.  Returns the path."""
    import numpy as np

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        from PIL import Image
        mode = "RGB" if img.shape[2] == 3 else "RGBA"
        Image.fromarray(np.ascontiguousarray(img), mode).save(path)
        return path
    except Exception:
        h, w, _ = img.shape
        with open(path, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(np.ascontiguousarray(img[..., :3]).tobytes())
        return path


def report(quiet: bool = False) -> None:
    """Print a short environment / backend report."""
    import triangle_rasterizer as tr
    if quiet:
        return
    print(f"backend           : {tr.backend()}")
    print(f"C library version : {tr.version()}")
    print(f"OpenMP available  : {tr.omp_available()}")
    print(f"OpenMP max threads: {tr.omp_max_threads()}")
    print()
