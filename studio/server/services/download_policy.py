"""Select installed Hub handlers without installing dependencies or starting downloads."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.services.tool_jobs import ToolJobPaths


def apply_download_policy(
    paths: ToolJobPaths,
    mode: str,
    *,
    module_available: Callable[[str], bool] | None = None,
) -> ToolJobPaths:
    if mode not in {"auto", "disabled", "enabled"}:
        raise ValueError("download policy must be auto, disabled, or enabled")
    if mode == "disabled":
        return replace(paths, modelscope=None, huggingface=None, internal_modelscope=False, internal_huggingface=False)
    available = module_available or (lambda name: importlib.util.find_spec(name) is not None)
    selected = replace(
        paths,
        modelscope=paths.modelscope if paths.modelscope is not None and paths.modelscope.is_file() else None,
        huggingface=paths.huggingface if paths.huggingface is not None and paths.huggingface.is_file() else None,
        internal_modelscope=available("modelscope_hub"),
        internal_huggingface=available("huggingface_hub"),
    )
    if mode == "enabled" and not any((selected.modelscope, selected.huggingface, selected.internal_modelscope, selected.internal_huggingface)):
        raise ValueError("download handlers were enabled but no supported Hub SDK or executable is installed")
    return selected
