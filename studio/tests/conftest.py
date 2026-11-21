"""Portable execution of Python runtime fixtures and filesystem capabilities."""

import asyncio
import os
from pathlib import Path
import sys

import pytest

# The Agent host launcher lives in the repository's own scripts/ directory; make it
# importable no matter which directory pytest is started from.
sys.path.append(str(Path(__file__).resolve().parents[2]))


@pytest.fixture(autouse=True)
def windows_test_capabilities(tmp_path_factory):
    if os.name != "nt":
        yield
        return
    root = tmp_path_factory.getbasetemp()
    launch = asyncio.create_subprocess_exec
    symlink = Path.symlink_to

    async def launch_script(*command, **kwargs):
        if command and isinstance(command[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(command[0]))
            if path.is_absolute() and path.is_relative_to(root) and path.is_file():
                with path.open("rb") as source:
                    header = source.read(64).split(b"\n", 1)[0]
                if header.startswith(b"#!") and b"python" in header:
                    command = (sys.executable, *command)
        return await launch(*command, **kwargs)

    def create_symlink(path, target, target_is_directory=False):
        try:
            return symlink(path, target, target_is_directory=target_is_directory)
        except OSError as error:
            if path.parent != root and path.is_relative_to(root) and getattr(error, "winerror", None) == 1314:
                pytest.skip("symlink creation requires Windows developer mode or privilege")
            raise

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(asyncio, "create_subprocess_exec", launch_script)
        patch.setattr(Path, "symlink_to", create_symlink)
        yield
