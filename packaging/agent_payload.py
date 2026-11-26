"""Collect and precompress the Windows Agent's installed resources for NSIS."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import zipfile


def agent_payload(project: Path) -> dict[str, Path]:
    tauri = project / "app/src-tauri"
    bundle = json.loads((tauri / "tauri.release-windows.conf.json").read_text())["bundle"]
    resources = dict(bundle["resources"])
    resources.pop("../web/dist", None)
    resources["agent/runtime/"] = "agent-runtime"
    files: dict[str, Path] = {}
    names: set[str] = set()

    def add(source: Path, destination: str) -> None:
        path = PurePosixPath(destination.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in str(path) or any(c in str(path) for c in "\r\n\0"):
            raise ValueError(f"Unsafe Agent archive path: {destination}")
        key = str(path).casefold()
        if key in names:
            raise ValueError(f"Duplicate Agent archive path: {path}")
        if not source.is_file():
            raise FileNotFoundError(source)
        names.add(key)
        files[str(path)] = source

    for source, destination in resources.items():
        root = (tauri / source).resolve()
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    add(path, str(PurePosixPath(destination) / path.relative_to(root).as_posix()))
        else:
            add(root, destination)
    for source in bundle["externalBin"]:
        add((tauri / f"{source}-x86_64-pc-windows-msvc.exe").resolve(), Path(source).name + ".exe")
    for required in ("agent-runtime/node.exe", "agent-runtime/boot.mjs",
                     "agent-runtime/node_modules/@tyloinfer/dsh-plugin/package.json",
                     "tyloinfer-server/tyloinfer-server.exe", "mfq-runtime.exe"):
        if required not in files:
            raise FileNotFoundError(f"Missing Windows Agent resource: {required}")
    return files


def _nsis_string(value: str) -> str:
    return value.replace("$", "$$").replace('"', '$\\"')


def build_agent_payload(project: Path, archive: Path) -> Path:
    files = agent_payload(project)
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".zip.partial")
    uninstall = archive.with_name("agent-payload-uninstall.nsh")
    try:
        # NSIS solid compressors retain an uncompressed data block with a 2 GiB
        # limit. A precompressed ZIP keeps that block small without dropping files.
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
            for name, path in sorted(files.items()):
                package.write(path, name)
        roots = sorted({PurePosixPath(name).parts[0] for name in files})
        directories = {PurePosixPath(name).parts[0] for name in files if len(PurePosixPath(name).parts) > 1}
        expanded_kib = (sum(entry.file_size for entry in package.infolist()) + 1023) // 1024
        instructions = ["; Generated from the archive: remove only installed payload paths.",
                        f"!define TYLOINFER_AGENT_PAYLOAD_KIB {expanded_kib}",
                        "!macro TYLOINFER_AGENT_REMOVE_PAYLOAD"]
        for name in roots:
            command = "RMDir /r" if name in directories else "Delete"
            instructions.append(f'  {command} "$INSTDIR\\{_nsis_string(name)}"')
        instructions.append("!macroend")
        uninstall.write_text("\n".join(instructions) + "\n", encoding="utf-8")
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Packed {len(files)} Windows Agent resources into {archive} "
          f"({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    return archive
