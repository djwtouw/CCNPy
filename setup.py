import os
import setuptools
from distutils.core import Extension, setup

import pybind11

if r"MSC" in pybind11.sys.version:
    cpp_args = ["/std:c++17", "-UNDEBUG", "/Ox"]
else:
    cpp_args = ["-std=c++17", "-UNDEBUG", "-O3"]

package_name = "ccnpy"

ext_modules = [
    Extension(
        f"{package_name}._{package_name}",
        ["cpp/src/" + file for file in os.listdir("cpp/src")],
        include_dirs=[
            "pybind11/include", "cpp/include", pybind11.get_include()
        ],
        language="c++",
        extra_compile_args=cpp_args,
    ),
]

setup(
    ext_modules=ext_modules,
    packages=setuptools.find_packages(),
    zip_safe=False
)
