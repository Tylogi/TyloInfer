"""Build the portable CPU row fitter; no GPU architecture is required."""

import sys

from setuptools import Extension, find_packages, setup

setup(
    packages=find_packages(exclude=["tests", "tests.*"]),
    py_modules=["config", "serve", "_version"],
    ext_modules=[
        Extension(
            "_gyro_fit",
            sources=["_gyro/module.c", "_gyro/codebook.c"],
            depends=["_gyro/fit_core.inc", "_gyro/gyro_internal.h"],
            libraries=[] if sys.platform == "win32" else ["m"],
        )
    ],
)
