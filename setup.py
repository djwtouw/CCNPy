import os
import pybind11
import setuptools
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class DownloadEigen(build_ext):
    def run(self):
        eigen_dir = os.path.join(self.build_temp, "eigen-src")
        if not os.path.exists(eigen_dir):
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "--branch", "3.4.0",
                "https://gitlab.com/libeigen/eigen.git", eigen_dir
            ])

        # Destination in build tree
        include_dst = os.path.join(self.build_temp, "cpp", "include", "Eigen")
        eigen_include_src = os.path.join(eigen_dir, "Eigen")
        os.makedirs(os.path.dirname(include_dst), exist_ok=True)

        # Copy Eigen headers
        if os.path.exists(include_dst):
            shutil.rmtree(include_dst)
        shutil.copytree(eigen_include_src, include_dst)

        # Set include dirs
        for ext in self.extensions:
            ext.include_dirs.append(os.path.join(eigen_dir))
        super().run()


if r"MSC" in pybind11.sys.version:
    # cpp_args = ["/std:c++17", "/DEBUG"] # Debug
    cpp_args = ["/std:c++17", "/NDEBUG", "/Ox"] # Release
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
    cmdclass={"build_ext": DownloadEigen},
    packages=setuptools.find_packages(),
    zip_safe=False
)
