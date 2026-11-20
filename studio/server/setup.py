"""Build the portable CPU row fitter; no GPU architecture is required."""

import sys

from setuptools import Extension, find_packages, setup

# This project root is the `server` package, so every found package is reparented under it.
packages = ["server." + name for name in find_packages(".") if not name.split(".")[0] == "tests"]

setup(
    packages=packages,
    package_dir={"server": "."},
    ext_modules=[
        Extension(
            "server._gyro_fit",
            sources=["_gyro/module.c", "_gyro/codebook.c"],
            depends=["_gyro/fit_core.inc", "_gyro/gyro_internal.h"],
            define_macros=[],
            libraries=[] if sys.platform == "win32" else ["m"],
        )
    ],
)
