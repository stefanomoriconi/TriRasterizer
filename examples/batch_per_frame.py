"""Batch (repeated rendering) example.

A :class:`TriangleBatch` holds a fixed set of triangles on the C side and
can be re-rendered into any number of buffers with a single ``draw`` call —
ideal when the same scene is rendered many times (per-frame, per-cam,
per-resolution) without re-parsing the triangle list in Python.

This example:

  * builds one batch with 6 triangles;
  * renders the *same* batch into 3 differently-sized / coloured buffers;
  * shows how to ``add`` a new triangle at runtime and re-render.

Output: ``examples/output/batch_{small,medium,large}.png`` and
``examples/output/batch_grown.png``.

Run:  python examples/batch_per_frame.py
"""

import os

import numpy as np

import common  # noqa: F401
from common import ensure_output_dir, report, save_image

import triangle_rasterizer as tr


def base_scene() -> list:
    return [
        dict(v0=(10, 10),   v1=(90, 10),   v2=(50, 90),  color=(220, 40, 40),   alpha=1.0),
        dict(v0=(60, 20),   v1=(140, 20),  v2=(100, 100), color=(40, 120, 230), alpha=1.0),
        dict(v0=(30, 60),   v1=(110, 60),  v2=(70, 140),  color=(40, 200, 90),  alpha=0.7),
        dict(v0=(80, 90),   v1=(160, 90),  v2=(120, 170), color=(255, 220, 60), alpha=0.8),
    ]


def main() -> int:
    report()

    with tr.TriangleBatch(base_scene()) as batch:
        print(f"initial batch count: {batch.count}")

        # Render the same batch into three differently sized buffers.
        for name, size, bg in (
            ("small", 96, (10, 12, 16)),
            ("medium", 192, (18, 20, 26)),
            ("large", 384, (26, 28, 34)),
        ):
            img = np.full((size, size, 3), bg, dtype=np.uint8)
            batch.draw(img, n_threads=0)
            out = os.path.join(ensure_output_dir(), f"batch_{name}.png")
            save_image(img, out)
            covered = int(np.count_nonzero(np.any(img != np.array(bg, dtype=np.uint8), axis=2)))
            print(f"  {name:>6}  {size}x{size}  covered={covered}/{size*size}  -> {out}")

        # Grow the batch at runtime and re-render into one buffer.
        batch.add(dict(v0=(20, 20), v1=(180, 20), v2=(100, 200),
                       color=(200, 80, 220), alpha=0.6))
        print(f"after add         : {batch.count}")
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        batch.draw(img, n_threads=0)
        out = os.path.join(ensure_output_dir(), "batch_grown.png")
        save_image(img, out)
        print(f"  grown           256x256  -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
