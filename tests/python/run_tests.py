"""Dependency-free standalone runner for the Python test suite.

This exists because the target environment has no network access, so
``pytest`` may not be installable.  It collects every ``test_*`` callable
from ``test_triangle_rasterizer.py`` and runs each one, printing a PASS/FAIL
line per test and a final summary.  The exit code is ``0`` when all tests
pass and ``1`` otherwise, so it can be used directly in CI.

Usage:
    python tests/python/run_tests.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

TEST_MODULE_FILE = os.path.join(_HERE, "test_triangle_rasterizer.py")


def _load_test_module():
    spec = importlib.util.spec_from_file_location("test_triangle_rasterizer", TEST_MODULE_FILE)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError("could not create import spec for %s" % TEST_MODULE_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_test_module()

    # `pytest.skip()` (used by test_cuda_parity_if_available for non-CUDA
    # builds) raises pytest's `Skipped`, which subclasses `BaseException`
    # (not `Exception`) -- so a plain `except Exception` does not catch it.
    # Detect it (if pytest is importable) so a skip is reported and handled
    # gracefully instead of crashing this whole standalone runner.
    try:
        from _pytest.outcomes import Skipped as _Skipped
    except Exception:  # pytest not installed, or internals changed
        _Skipped = ()

    tests = []
    for name in sorted(vars(module)):
        if name.startswith("test_"):
            obj = getattr(module, name)
            if callable(obj):
                tests.append((name, obj))

    if not tests:
        print("[FAIL] no tests discovered in %s" % TEST_MODULE_FILE)
        return 1

    print("Discovered %d test(s) in %s" % (len(tests), os.path.basename(TEST_MODULE_FILE)))
    print("-" * 64)

    failures = []
    skipped = []
    for name, fn in tests:
        try:
            fn()
            print("PASS  %s" % name)
        except _Skipped as e:
            skipped.append(name)
            print("SKIP  %s (%s)" % (name, e))
        except Exception:
            failures.append(name)
            print("FAIL  %s" % name)
            traceback.print_exc()

    print("-" * 64)
    if failures:
        print("[FAIL] %d/%d tests failed: %s" % (len(failures), len(tests), ", ".join(failures)))
        return 1
    suffix = (" (%d skipped)" % len(skipped)) if skipped else ""
    print("[PASS] %d/%d python tests passed%s" % (len(tests) - len(skipped), len(tests), suffix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
