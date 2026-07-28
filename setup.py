"""Build the CCNPy C++ extension (ccnpy._core).

The algorithm lives in the framework-agnostic core under ``ccncpp/``. This file
only compiles that core together with the thin pybind11 binding in
``src/ccnpy/_binding.cpp``.

Eigen is located, in order, from:

1. the ``EIGEN_INCLUDE_DIR`` environment variable,
2. a vendored copy at ``extern/eigen`` (e.g. a git submodule),
3. common system locations.

The directory must be the one that *contains* the ``Eigen/`` header folder.
"""

import os
import sys
from glob import glob

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

HERE = os.path.abspath(os.path.dirname(__file__))


def find_eigen() -> str:
    candidates = []
    env = os.environ.get("EIGEN_INCLUDE_DIR")
    if env:
        candidates.append(env)
    candidates.append(os.path.join(HERE, "extern", "eigen"))
    candidates += [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "/opt/homebrew/include/eigen3",
    ]
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "Eigen", "Dense")):
            return path
    raise RuntimeError(
        "Could not locate Eigen. Set EIGEN_INCLUDE_DIR to the directory "
        "containing the 'Eigen/' header folder, vendor it at extern/eigen "
        "(e.g. `git submodule update --init`), or install it system-wide."
    )


core_sources = sorted(glob(os.path.join("ccncpp", "src", "*.cpp")))
binding_sources = [os.path.join("src", "ccnpy", "_binding.cpp")]

ext_modules = [
    Pybind11Extension(
        "ccnpy._core",
        sources=binding_sources + core_sources,
        include_dirs=[os.path.join("ccncpp", "include"), find_eigen()],
        cxx_std=17,
        define_macros=[("NDEBUG", "1")],
    )
]

# Optional release optimization flags (pybind11 already sets a sane baseline).
if not sys.platform.startswith("win"):
    for ext in ext_modules:
        ext.extra_compile_args = (ext.extra_compile_args or []) + ["-O3"]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
