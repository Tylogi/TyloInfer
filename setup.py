"""Build the portable CPU row fitter; no GPU architecture is required."""

import sys
import runpy
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        sources = runpy.run_path(str(root / "packaging/python/build.py"))
        files = sources["cuda_files"](root)
        sources["collect_cuda_sources"](root, Path(self.build_lib), files)
        sources["collect_tq_sources"](root, Path(self.build_lib))

setup(
    cmdclass={"build_py": BuildPy},
    ext_modules=[
        Extension(
            "tyloinfer._gyro_fit",
            sources=["tyloinfer/_gyro/module.c", "tyloinfer/_gyro/codebook.c"],
            depends=["tyloinfer/_gyro/fit_core.inc", "tyloinfer/_gyro/gyro_internal.h"],
            define_macros=[],
            libraries=[] if sys.platform == "win32" else ["m"],
        )
    ]
)
