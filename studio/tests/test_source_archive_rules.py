from pathlib import Path
import shutil
import subprocess

import pytest


def test_downloaded_weights_stay_visible_and_model_source_dirs_are_not_ignored(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is required to check source archive rules")
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text((root / ".gitignore").read_text())
    # Weights are deliberately not ignored (see CONTRIBUTING.md): an accidental
    # commit must show up in git status. What must never happen is an unanchored
    # rule that hides a source directory named models/.
    for path, ignored in (
        ("models/local/checkpoint.tq", False),
        ("weights/giant.gguf", False),
        ("build/checkpoint.tq", True),
        ("csrc/models/model.cpp", False),
        ("csrc/backends/metal/models/deepseek_v41/model.cpp", False),
        ("csrc/backends/cuda/models/qwen35/model.cpp", False),
        ("csrc/backends/hip/models/qwen35/model.cpp", False),
    ):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", path],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert result.returncode in (0, 1), result.stderr
        assert (result.returncode == 0) is ignored, path
