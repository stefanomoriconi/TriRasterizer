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
    for name, fn in tests:
        try:
            fn()
            print("PASS  %s" % name)
        except Exception:
            failures.append(name)
            print("FAIL  %s" % name)
            traceback.print_exc()

    print("-" * 64)
    if failures:
        print("[FAIL] %d/%d tests failed: %s" % (len(failures), len(tests), ", ".join(failures)))
        return 1
    print("[PASS] %d/%d python tests passed" % (len(tests), len(tests)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
