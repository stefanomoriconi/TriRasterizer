"""Library discovery, loading and ctypes prototype declaration.

This module locates and loads the compiled ``triangle_rasterizer`` shared
library and declares the C ABI signatures.  It is an internal module; use
the public API in :mod:`triangle_rasterizer` instead.

Discovery order
---------------
1. ``TRI_RASTERIZER_DLL`` environment variable (explicit full path).
2. Common CMake build output folders relative to the repository root
   (``build/Release``, ``build/Debug``, ``build``).
3. Next to this package (in case the library was copied/bundled in).
4. The repository root.
5. System search (``PATH`` / ``LD_LIBRARY_PATH`` / ``DYLD_*``).

The same name is used on all platforms: ``triangle_rasterizer.dll`` on
Windows, ``libtriangle_rasterizer.dylib`` on macOS,
``libtriangle_rasterizer.so`` on Linux.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from typing import List, Optional


def _lib_names() -> List[str]:
    if sys.platform.startswith("win"):
        return ["triangle_rasterizer.dll"]
    if sys.platform == "darwin":
        return ["libtriangle_rasterizer.dylib"]
    return ["libtriangle_rasterizer.so"]


def candidate_lib_paths() -> List[str]:
    """Return candidate library file paths, in discovery order."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)  # repository root (package is at <root>/triangle_rasterizer/)
    names = _lib_names()
    candidates: List[str] = []

    env = os.environ.get("TRI_RASTERIZER_DLL")
    if env:
        candidates.append(env)

    search_dirs = [
        os.path.join(root, "build", "Release"),
        os.path.join(root, "build", "Debug"),
        os.path.join(root, "build"),
        here,
        root,
    ]
    for base in search_dirs:
        base = os.path.normpath(base)
        candidates.extend(os.path.join(base, n) for n in names)

    sys_name = ctypes.util.find_library("triangle_rasterizer")
    if sys_name:
        candidates.append(sys_name)
    return candidates


def load_library():
    """Load the shared library, raising a helpful ImportError if not found."""
    last_err: Optional[OSError] = None
    for path in candidate_lib_paths():
        try:
            if sys.platform.startswith("win"):
                return ctypes.WinDLL(path)
            return ctypes.CDLL(path)
        except OSError as e:  # file missing / cannot load
            last_err = e
    raise ImportError(
        "Could not load the triangle_rasterizer shared library. "
        "Build it first (see README, e.g. cmake -B build && cmake --build build) "
        "or set TRI_RASTERIZER_DLL to the full path of the library file.\n"
        "Last error: " + repr(last_err)
    )


# ctypes struct mirroring the C `RasterTriangle` (10 floats, 40 bytes).
class RasterTriangle(ctypes.Structure):
    _fields_ = [
        ("v0", ctypes.c_float * 2),
        ("v1", ctypes.c_float * 2),
        ("v2", ctypes.c_float * 2),
        ("color", ctypes.c_float * 3),
        ("alpha", ctypes.c_float),
    ]


def declare_prototypes(lib) -> None:
    """Set ``argtypes``/``restype`` for every C symbol we call."""
    c_float = ctypes.c_float  # noqa: F841 (kept for clarity)
    c_uint8 = ctypes.c_uint8
    c_int = ctypes.c_int
    c_size_t = ctypes.c_size_t
    c_void_p = ctypes.c_void_p
    c_char_p = ctypes.c_char_p
    p_uint8 = ctypes.POINTER(c_uint8)
    p_tri = ctypes.POINTER(RasterTriangle)

    # Single triangle
    f = lib.rasterize_triangle
    f.argtypes = [p_uint8, c_int, c_int, c_int, p_tri, c_int]
    f.restype = c_int

    # Array of triangles
    f = lib.rasterize_triangles
    f.argtypes = [p_uint8, c_int, c_int, c_int, p_tri, c_size_t, c_int]
    f.restype = c_int

    # Batch handle creation / query / destruction
    f = lib.rasterizer_batch_from_triangles
    f.argtypes = [p_tri, c_size_t]
    f.restype = c_void_p

    f = lib.rasterizer_batch_new
    f.argtypes = [c_size_t]
    f.restype = c_void_p

    f = lib.rasterizer_batch_add
    f.argtypes = [c_void_p, p_tri]
    f.restype = c_int

    f = lib.rasterizer_batch_count
    f.argtypes = [c_void_p]
    f.restype = c_size_t

    f = lib.rasterizer_batch_free
    f.argtypes = [c_void_p]
    f.restype = None

    f = lib.rasterize_triangles_batch
    f.argtypes = [p_uint8, c_int, c_int, c_int, c_void_p, c_int]
    f.restype = c_int

    # Anti-aliasing overloads (v1.2.0) — same as above plus trailing ``int aa``.
    f = lib.rasterize_triangles_ex
    f.argtypes = [p_uint8, c_int, c_int, c_int, p_tri, c_size_t, c_int, c_int]
    f.restype = c_int

    f = lib.rasterize_triangles_batch_ex
    f.argtypes = [p_uint8, c_int, c_int, c_int, c_void_p, c_int, c_int]
    f.restype = c_int

    f = lib.rasterize_triangle_ex
    f.argtypes = [p_uint8, c_int, c_int, c_int, p_tri, c_int, c_int]
    f.restype = c_int

    # Precision overloads (v1.3.0) — same as `_ex` plus trailing ``int precision``.
    f = lib.rasterize_triangles_v2
    f.argtypes = [p_uint8, c_int, c_int, c_int, p_tri, c_size_t, c_int, c_int, c_int]
    f.restype = c_int

    f = lib.rasterize_triangles_batch_v2
    f.argtypes = [p_uint8, c_int, c_int, c_int, c_void_p, c_int, c_int, c_int]
    f.restype = c_int

    f = lib.rasterize_triangle_v2
    f.argtypes = [p_uint8, c_int, c_int, c_int, p_tri, c_int, c_int, c_int]
    f.restype = c_int

    # Diagnostics
    f = lib.rasterizer_last_error
    f.argtypes = []
    f.restype = c_int

    f = lib.rasterizer_error_string
    f.argtypes = [c_int]
    f.restype = c_char_p

    f = lib.rasterizer_version
    f.argtypes = []
    f.restype = c_char_p

    f = lib.rasterizer_omp_available
    f.argtypes = []
    f.restype = c_int

    f = lib.rasterizer_omp_max_threads
    f.argtypes = []
    f.restype = c_int

    f = lib.rasterizer_backend
    f.argtypes = []
    f.restype = c_char_p


_lib = load_library()
declare_prototypes(_lib)
