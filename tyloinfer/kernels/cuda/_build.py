"""Prepare canonical sources for Torch's flat object-file namespace."""

from __future__ import annotations

import hashlib
from pathlib import Path


def stage_sources(sources: list[str], build_directory: Path) -> list[str]:
    directory = build_directory / "sources"
    directory.mkdir(parents=True, exist_ok=True)
    staged = []
    for index, name in enumerate(sources):
        source = Path(name).resolve()
        target = directory / f"{index}_{source.name}"
        # Torch hashes entry files; include the canonical content hash to trigger rebuilds.
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        content = f'// {digest}\n#include "{source.as_posix()}"\n'
        if not target.is_file() or target.read_text(encoding="utf-8") != content:
            target.write_text(content, encoding="utf-8")
        staged.append(str(target))
    return staged
