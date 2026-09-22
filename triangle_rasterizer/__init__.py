"""
triangle_rasterizer — Python API for the C/CUDA triangle rasterizer.
====================================================================

This package exposes a clean, NumPy-friendly API over the stable C ABI
declared in ``include/triangle_rasterizer.h``.  The underlying shared
library can be built with a CPU (OpenMP) backend or a CUDA backend;
whichever is available is selected at library build time and reported
by :func:`backend`.

Example
-------
    import numpy as np
    import triangle_rasterizer as tr

    img = np.zeros((256, 256, 3), dtype=np.uint8)
    triangles = [
        # vertices are (x, y) in pixel coordinates, colour is 0..255 RGB
        dict(v0=(40, 40),  v1=(216, 40),  v2=(128, 216), color=(255, 0, 0),   alpha=1.0),
        dict(v0=(60, 80),  v1=(196, 80),  v2=(128, 208), color=(0, 200, 255), alpha=0.6),
    ]
    tr.rasterize(img, triangles, n_threads=0)   # 0 -> let OpenMP decide
    # img now contains the rendered triangles (mutated in place).

Colour convention
-----------------
RGB components are 0..255 (int or float).  Alpha is 0..1.  The image is
a C-contiguous ``uint8`` array of shape ``(H, W, 3)`` or ``(H, W, 4)``;
it is mutated in place and also returned.

Thread control
--------------
``n_threads=0`` (default) lets the library use its default thread count
(OpenMP on the CPU backend).  Pass ``n >= 1`` to cap the number of
threads.  Every scanline / pixel is independent, so results are identical
regardless of the thread count — and bit-identical between the CPU and
CUDA backends.

Backends
--------
Call :func:`backend` to see which backend the loaded library was built
with (``"cuda"`` when a CUDA device is present, otherwise ``"cpu"`` /
``"cpu (OpenMP)"``).  The Python API is the same for both.

Anti-aliasing (v1.2.0)
----------------------
Every render call accepts an ``antialiasing`` keyword (default ``AA_NONE``):

    img = np.zeros((256, 256, 3), dtype=np.uint8)
    tr.rasterize(img, triangles, antialiasing=tr.AA_ANALYTICAL)

``AA_NONE`` is the legacy 1.1.0 behaviour (hard edges, bit-identical).
``AA_ANALYTICAL`` computes the exact area coverage of each pixel, while
``AA_SSAA2X2`` / ``AA_SSAA4ROT`` use 4-tap supersampling.  Higher modes
smooth edges at a small cost; results remain deterministic and
thread-count independent.

Precision (v1.3.0)
-------------------
Every render call also accepts a ``precision`` keyword (default
``PRECISION_FLOAT``):

    tr.rasterize(img, triangles, precision=tr.PRECISION_FIXED8)

``PRECISION_FLOAT`` is the legacy double-precision scanline maths.
``PRECISION_FIXED8`` uses an OpenCV-style Q24.8 fixed-point scanline
(1/256 px sub-pixel resolution); output matches ``PRECISION_FLOAT`` to
within a fraction of a pixel (not bit-identical).
"""

from __future__ import annotations

import ctypes
from typing import Optional, Sequence, Tuple

import numpy as np

from ._loader import RasterTriangle, _lib

__all__ = [
    "RasterTriangle",
    "rasterize",
    "draw_triangle",
    "TriangleBatch",
    "backend",
    "version",
    "omp_available",
    "omp_max_threads",
    "last_error",
    "error_string",
    "RasterizerError",
    "lib",
    # Anti-aliasing modes (v1.2.0)
    "AA_NONE",
    "AA_ANALYTICAL",
    "AA_SSAA2X2",
    "AA_SSAA4ROT",
    # Precision modes (v1.3.0)
    "PRECISION_FLOAT",
    "PRECISION_FIXED8",
]

__version__ = "1.3.0"


# --------------------------------------------------------------------------- #
# Anti-aliasing modes (v1.2.0).  Mirrors the ``TRI_AA_*`` C enum.
#   AA_NONE        — legacy per-pixel inside test, hard edges (default).
#   AA_ANALYTICAL  — coverage = fraction of the pixel area inside the triangle.
#   AA_SSAA2X2     — 4 sub-pixel samples on a 2x2 grid (1/4,3/4).
#   AA_SSAA4ROT    — 4 sub-pixel samples on a rotated grid.
# --------------------------------------------------------------------------- #
AA_NONE = 0
AA_ANALYTICAL = 1
AA_SSAA2X2 = 2
AA_SSAA4ROT = 3

_AA_MODES = (AA_NONE, AA_ANALYTICAL, AA_SSAA2X2, AA_SSAA4ROT)


def _check_aa(aa: int) -> int:
    if not isinstance(aa, (int, np.integer)):
        raise TypeError(f"antialiasing must be an int AA_* mode; got {type(aa)!r}")
    aa = int(aa)
    if aa not in _AA_MODES:
        raise ValueError(
            f"antialiasing must be one of AA_NONE={AA_NONE}, "
            f"AA_ANALYTICAL={AA_ANALYTICAL}, AA_SSAA2X2={AA_SSAA2X2}, "
            f"AA_SSAA4ROT={AA_SSAA4ROT}; got {aa}"
        )
    return aa


# --------------------------------------------------------------------------- #
# Precision modes (v1.3.0).  Mirrors the ``TRI_PRECISION_*`` C enum.
#   PRECISION_FLOAT   — legacy double-precision scanline maths (default).
#   PRECISION_FIXED8  — Q24.8 fixed-point scanline maths (OpenCV-style).
# --------------------------------------------------------------------------- #
PRECISION_FLOAT = 0
PRECISION_FIXED8 = 1

_PRECISION_MODES = (PRECISION_FLOAT, PRECISION_FIXED8)


def _check_precision(precision: int) -> int:
    if not isinstance(precision, (int, np.integer)):
        raise TypeError(f"precision must be an int PRECISION_* mode; got {type(precision)!r}")
    precision = int(precision)
    if precision not in _PRECISION_MODES:
        raise ValueError(
            f"precision must be one of PRECISION_FLOAT={PRECISION_FLOAT}, "
            f"PRECISION_FIXED8={PRECISION_FIXED8}; got {precision}"
        )
    return precision


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
class RasterizerError(RuntimeError):
    """Raised when a call into the C library returns an error code."""

    def __init__(self, code: int):
        self.code = code
        # Best-effort decode of the message using the library itself.
        try:
            msg = _lib.rasterizer_error_string(code)
            msg = msg.decode("utf-8", "replace") if isinstance(msg, bytes) else str(msg)
        except Exception:  # pragma: no cover
            msg = "unknown error"
        super().__init__(f"triangle_rasterizer error {code}: {msg}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ensure_image(img: np.ndarray) -> np.ndarray:
    """Validate and normalise the target image (contiguous uint8 HxWx3|4)."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    if a.ndim == 2:
        # single-channel input is not supported by the library; expand to RGB
        a = np.dstack([a, a, a])
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"image must have shape (H, W) or (H, W, 3|4); got {a.shape}")
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    return a


def _normalise_triangle(spec) -> RasterTriangle:
    """Accept flexible Python triangle specs and fill a RasterTriangle struct."""
    if isinstance(spec, RasterTriangle):
        return spec

    # dict form: v0,v1,v2,color,alpha
    if isinstance(spec, dict):
        t = RasterTriangle()
        _fill_vertex(t, "v0", spec.get("v0", spec.get("p0")))
        _fill_vertex(t, "v1", spec.get("v1", spec.get("p1")))
        _fill_vertex(t, "v2", spec.get("v2", spec.get("p2")))
        _fill_color(t, spec.get("color", (255, 255, 255)))
        t.alpha = float(spec.get("alpha", 1.0))
        return t

    # tuple/list form: (v0, v1, v2[, color[, alpha]])
    if isinstance(spec, (tuple, list, np.ndarray)):
        spec = list(spec)
        if len(spec) not in (3, 4, 5):
            raise ValueError(
                "triangle tuple must be (v0,v1,v2[,color[,alpha]]) "
                f"with 3, 4 or 5 elements; got {len(spec)}"
            )
        t = RasterTriangle()
        _fill_vertex(t, "v0", spec[0])
        _fill_vertex(t, "v1", spec[1])
        _fill_vertex(t, "v2", spec[2])
        _fill_color(t, spec[3] if len(spec) >= 4 else (255, 255, 255))
        t.alpha = float(spec[4] if len(spec) >= 5 else 1.0)
        return t

    raise TypeError(f"cannot interpret triangle spec: {type(spec)!r}")


def _fill_vertex(t: RasterTriangle, field: str, xy) -> None:
    try:
        x, y = xy
    except (TypeError, ValueError):
        raise ValueError(f"vertex '{field}' must be a (x, y) pair; got {xy!r}")
    getattr(t, field)[0] = float(x)
    getattr(t, field)[1] = float(y)


def _fill_color(t: RasterTriangle, rgb) -> None:
    try:
        r, g, b = rgb
    except (TypeError, ValueError):
        raise ValueError(f"color must be a (R, G, B) triple; got {rgb!r}")

    def clamp(v: float) -> float:
        return 0.0 if v < 0.0 else (255.0 if v > 255.0 else v)

    t.color[0] = clamp(float(r))
    t.color[1] = clamp(float(g))
    t.color[2] = clamp(float(b))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def rasterize(image: np.ndarray, triangles: Sequence, n_threads: int = 0,
              antialiasing: int = AA_NONE, precision: int = PRECISION_FLOAT) -> np.ndarray:
    """Rasterise ``triangles`` into ``image`` (mutated in place and returned).

    Parameters
    ----------
    image : np.ndarray
        ``uint8`` array of shape ``(H, W, 3)`` or ``(H, W, 4)``.
    triangles : sequence
        Each element is a triangle, either a dict
        ``{"v0":(x,y), "v1":(x,y), "v2":(x,y), "color":(r,g,b), "alpha":a}``
        or a tuple ``((x0,y0),(x1,y1),(x2,y2),(r,g,b),a)``.
        Colours are 0..255; alpha is 0..1.  Triangles are painted in input
        order (painter's algorithm).
    n_threads : int, optional
        0 (default) uses the library's default thread count; ``n >= 1`` caps it.
    antialiasing : int, optional
        Anti-aliasing mode (v1.2.0): ``AA_NONE`` (default, legacy hard edges),
        ``AA_ANALYTICAL`` (per-pixel area coverage), ``AA_SSAA2X2`` or
        ``AA_SSAA4ROT`` (4-tap supersampling).  See the module docstring.
    precision : int, optional
        Scanline precision mode (v1.3.0): ``PRECISION_FLOAT`` (default,
        legacy double-precision maths) or ``PRECISION_FIXED8`` (Q24.8
        fixed-point maths).  See the module docstring.

    Returns
    -------
    np.ndarray
        The (possibly newly allocated, contiguous) image, now filled.
    """
    aa = _check_aa(antialiasing)
    prec = _check_precision(precision)
    img = _ensure_image(image)
    h, w, ch = img.shape

    if triangles is None:
        return img
    tris = [_normalise_triangle(t) for t in triangles]
    if not tris:
        return img

    arr = (RasterTriangle * len(tris))()
    for i, t in enumerate(tris):
        arr[i] = t

    buf = img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    if aa == AA_NONE and prec == PRECISION_FLOAT:
        rc = _lib.rasterize_triangles(buf, w, h, ch, arr, len(tris), int(n_threads))
    elif prec == PRECISION_FLOAT:
        rc = _lib.rasterize_triangles_ex(buf, w, h, ch, arr, len(tris),
                                         int(n_threads), aa)
    else:
        rc = _lib.rasterize_triangles_v2(buf, w, h, ch, arr, len(tris),
                                         int(n_threads), aa, prec)
    if rc != 0:
        raise RasterizerError(rc)
    return img


def draw_triangle(
    image: np.ndarray,
    v0, v1, v2,
    color=(255, 255, 255),
    alpha: float = 1.0,
    n_threads: int = 0,
    antialiasing: int = AA_NONE,
    precision: int = PRECISION_FLOAT,
) -> np.ndarray:
    """Convenience wrapper to draw a single triangle.

    Parameters mirror :func:`rasterize`; ``antialiasing`` accepts the
    ``AA_*`` constants (v1.2.0) and ``precision`` the ``PRECISION_*``
    constants (v1.3.0).
    """
    return rasterize(
        image,
        [dict(v0=v0, v1=v1, v2=v2, color=color, alpha=alpha)],
        n_threads=n_threads,
        antialiasing=antialiasing,
        precision=precision,
    )


class TriangleBatch:
    """A persistent batch of triangles, rasterisable repeatedly (e.g. per-frame).

    Useful when the triangle set is fixed and you re-render into different
    buffers or the same buffer every frame.

    Example
    -------
        with TriangleBatch(triangles) as batch:
            batch.draw(img, n_threads=0)
    """

    def __init__(self, triangles: Optional[Sequence] = None, capacity_hint: int = 0):
        if triangles is None:
            handle = _lib.rasterizer_batch_new(int(capacity_hint))
        else:
            tris = [_normalise_triangle(t) for t in triangles]
            if not tris:
                raise ValueError("batch requires at least one triangle")
            arr = (RasterTriangle * len(tris))()
            for i, t in enumerate(tris):
                arr[i] = t
            handle = _lib.rasterizer_batch_from_triangles(arr, len(tris))
        if not handle:
            raise RasterizerError(_lib.rasterizer_last_error())
        self._handle = ctypes.c_void_p(handle)

    # -- context manager -----------------------------------------------------
    def __enter__(self) -> "TriangleBatch":
        return self

    def __exit__(self, *exc) -> None:
        self.free()

    # -- mutations -----------------------------------------------------------
    def add(self, triangle) -> None:
        t = _normalise_triangle(triangle)
        rc = _lib.rasterizer_batch_add(self._handle, t)
        if rc != 0:
            raise RasterizerError(rc)

    # -- queries -------------------------------------------------------------
    @property
    def count(self) -> int:
        return int(_lib.rasterizer_batch_count(self._handle))

    # -- rendering -----------------------------------------------------------
    def draw(self, image: np.ndarray, n_threads: int = 0,
             antialiasing: int = AA_NONE, precision: int = PRECISION_FLOAT) -> np.ndarray:
        """Render the batch into ``image``.

        ``antialiasing`` (v1.2.0) accepts the ``AA_*`` constants and
        ``precision`` (v1.3.0) the ``PRECISION_*`` constants; the defaults
        (``AA_NONE``, ``PRECISION_FLOAT``) reproduce the legacy 1.1.0
        behaviour bit-for-bit.
        """
        aa = _check_aa(antialiasing)
        prec = _check_precision(precision)
        img = _ensure_image(image)
        h, w, ch = img.shape
        buf = img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        if aa == AA_NONE and prec == PRECISION_FLOAT:
            rc = _lib.rasterize_triangles_batch(buf, w, h, ch, self._handle, int(n_threads))
        elif prec == PRECISION_FLOAT:
            rc = _lib.rasterize_triangles_batch_ex(buf, w, h, ch, self._handle,
                                                   int(n_threads), aa)
        else:
            rc = _lib.rasterize_triangles_batch_v2(buf, w, h, ch, self._handle,
                                                   int(n_threads), aa, prec)
        if rc != 0:
            raise RasterizerError(rc)
        return img

    def free(self) -> None:
        if self._handle and self._handle.value:
            _lib.rasterizer_batch_free(self._handle)
            self._handle = ctypes.c_void_p(None)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def backend() -> str:
    """Return the active rasterization backend name.

    ``"cuda"`` when the library was built with the CUDA backend and a CUDA
    device is available, otherwise a CPU identifier such as
    ``"cpu (OpenMP)"`` or ``"cpu"``.
    """
    s = _lib.rasterizer_backend()
    return s.decode("utf-8", "replace") if isinstance(s, bytes) else str(s)


def version() -> str:
    v = _lib.rasterizer_version()
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def omp_available() -> bool:
    return bool(_lib.rasterizer_omp_available())


def omp_max_threads() -> int:
    return int(_lib.rasterizer_omp_max_threads())


def last_error() -> int:
    return int(_lib.rasterizer_last_error())


def error_string(code: int) -> str:
    s = _lib.rasterizer_error_string(int(code))
    return s.decode("utf-8", "replace") if isinstance(s, bytes) else str(s)


def lib():
    """Return the loaded ctypes library handle (advanced use)."""
    return _lib
