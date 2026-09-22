"""CPU / CUDA parity example.

The CPU and CUDA backends use the *same* containment test, blend order and
byte conversion, so a given triangle list must render to a **bit-identical**
image on either backend.  This example:

  * reports which backend the loaded library was built with;
  * renders a scene once with 1 thread and once with the default thread
    count and asserts the two are bit-identical (determinism);
  * prints a parity summary so you can compare against a CUDA build.

When you run the same scene on a machine with a CUDA build and a machine
with a CPU build, the two PNGs should be byte-for-byte identical (modulo
the PNG encoder).  This example writes ``parity_scene.png``.

Run:  python examples/cpu_vs_cuda_parity.py
"""

import os

import numpy as np

import common  # noqa: F401
from common import ensure_output_dir, report, save_image

import triangle_rasterizer as tr


def scene() -> list:
    rng = np.random.default_rng(42)
    tris = []
    for _ in range(40):
        pts = rng.integers(0, 256, size=(3, 2)).tolist()
        col = tuple(int(c) for c in rng.integers(0, 256, size=3).tolist())
        tris.append(dict(v0=pts[0], v1=pts[1], v2=pts[2],
                         color=col, alpha=float(rng.uniform(0.3, 1.0))))
    return tris


def main() -> int:
    report()

    W = H = 256
    tris = scene()

    img1 = np.zeros((H, W, 3), dtype=np.uint8)
    img2 = np.zeros((H, W, 3), dtype=np.uint8)
    tr.rasterize(img1, tris, n_threads=1)
    tr.rasterize(img2, tris, n_threads=0)

    identical = bool(np.array_equal(img1, img2))
    print(f"scene triangles   : {len(tris)}")
    print(f"1-thread vs N     : {'bit-identical' if identical else 'DIFFER'}")
    if not identical:
        print("!! determinism check FAILED")
        return 1

    out = os.path.join(ensure_output_dir(), "parity_scene.png")
    save_image(img2, out)
    print(f"saved             : {out}")
    print()
    print("To compare CPU vs CUDA: build twice (-DTRIANGLE_RASTERIZER_BACKEND=CPU / CUDA),")
    print("run this on both machines, and diff the two parity_scene.png files — they")
    print("should be identical because both backends use the same math and order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
