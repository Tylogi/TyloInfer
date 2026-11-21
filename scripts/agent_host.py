"""Launch the complete official DSH Host without importing the inference core."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

DSH_VERSION = "0.2.2-rc.2"
PROFILE = "tyloinfer-web"
OFFICIAL = ("@deepseek-ai/dsh", "@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app")
PRODUCTS = {"@tyloinfer/dsh-host": "host", "@tyloinfer/dsh-studio": "studio"}
BUNDLES = [*OFFICIAL[1:], *PRODUCTS]


class LauncherError(RuntimeError):
    """An unavailable or incompatible deployment, before process launch."""


def _read_manifest(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LauncherError(f"Missing or invalid manifest for {label}") from error
    if not isinstance(value, dict):
        raise LauncherError(f"Invalid manifest for {label}")
    return value


def runtime_version(runtime_root: Path) -> str:
    manifest = _read_manifest(runtime_root / "node_modules" / OFFICIAL[0] / "package.json", OFFICIAL[0])
    version = manifest.get("version")
    if manifest.get("name") != OFFICIAL[0] or not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?", version):
        raise LauncherError("The selected runtime has no supported DSH release identity")
    return version


def check_layout(runtime_root: Path, plugins_root: Path) -> Path:
    """Validate declared plugin compatibility and complete runtime package alignment."""
    version = runtime_version(runtime_root)
    for name in OFFICIAL:
        manifest = _read_manifest(runtime_root / "node_modules" / name / "package.json", name)
        if manifest.get("name") != name or manifest.get("version") != version:
            raise LauncherError(f"{name} must match the supported DSH {version}")
    for name, directory in PRODUCTS.items():
        root = plugins_root / directory
        manifest = _read_manifest(root / "package.json", name)
        if manifest.get("name") != name or manifest.get("engines", {}).get("dsh") != version:
            raise LauncherError(f"{name} must declare the supported DSH {version}")
        exports = manifest.get("exports", {})
        for entry in ([".", "./client"] if name.endswith("dsh-studio") else ["."]):
            target = exports.get(entry)
            if not isinstance(target, str) or not target.startswith("./"):
                raise LauncherError(f"Invalid {entry} entry for {name}")
            artifact = (root / target).resolve()
            if root.resolve() not in artifact.parents or not artifact.is_file():
                raise LauncherError(f"Missing built {entry} entry for {name}; build outside timing windows")
    carrier = runtime_root / "node_modules" / ".bin" / ("dsh.cmd" if os.name == "nt" else "dsh")
    if not carrier.is_file():
        raise LauncherError("The selected runtime has no official dsh CLI entry")
    return carrier


def isolated_environment(state_root: Path, inherited: dict[str, str]) -> dict[str, str]:
    """Use only this deployment's profile, workspace, temporary files and credentials."""
    if inherited.get("DSH_OFFICE_SIDECAR"):
        raise LauncherError("Inherited Office resources are not permitted")
    execution_keys = {
        "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "OS",
        "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432", "NUMBER_OF_PROCESSORS",
        "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "COLORTERM", "NO_COLOR",
    }
    environment = {key: value for key, value in inherited.items() if key.upper() in execution_keys}
    home, workspace = state_root / "home", state_root / "workspace"
    environment.update({
        "DSH_HOME": str(home), "HOME": str(workspace), "USERPROFILE": str(workspace),
        "APPDATA": str(home / "cache"), "LOCALAPPDATA": str(home / "cache"),
        "XDG_CONFIG_HOME": str(home / "cache"), "DSH_RESOURCE_CACHE": str(home / "cache"),
        "TEMP": str(home / "tmp"), "TMP": str(home / "tmp"),
        "HTTP_PROXY": "", "HTTPS_PROXY": "", "ALL_PROXY": "", "NO_PROXY": "127.0.0.1,localhost,::1",
        "NODE_OPTIONS": "", "NODE_PATH": "", "DSH_TELEMETRY_DISABLED": "1",
    })
    return environment


def check_profile_bindings(state_root: Path, plugins_root: Path) -> None:
    """Reject conflicting installed links before creating or writing profile files."""
    profile = state_root / "home" / "profiles" / PROFILE
    for name, directory in PRODUCTS.items():
        link = profile / "node_modules" / name
        if not os.path.lexists(link):
            continue
        try:
            matching = link.resolve() == (plugins_root / directory).resolve()
        except (OSError, RuntimeError) as error:
            raise LauncherError("Cannot resolve an existing product package binding") from error
        if not matching:
            raise LauncherError("An existing product package link points to a different installation")


def _link_directory(link: Path, target: Path) -> None:
    if os.path.lexists(link):
        if link.resolve() != target.resolve():
            raise LauncherError("An existing product package link points to a different installation")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        link.symlink_to(target, target_is_directory=True)
        return
    try:
        from _winapi import CreateJunction

        CreateJunction(str(target), str(link))
    except (ImportError, OSError) as error:
        raise LauncherError("Windows directory junction support is unavailable") from error


def prepare_profile(state_root: Path, plugins_root: Path, core_url: str) -> None:
    """Seed a complete default composition once; never overwrite saved user settings."""
    try:
        endpoint = urlsplit(core_url)
    except ValueError as error:
        raise LauncherError("The initial core endpoint is invalid") from error
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or endpoint.password:
        raise LauncherError("The initial core endpoint must be HTTP(S) without embedded credentials")
    if endpoint.query or endpoint.fragment:
        raise LauncherError("The initial core endpoint cannot contain a query or fragment")
    check_profile_bindings(state_root, plugins_root)
    home = state_root / "home"
    profile = home / "profiles" / PROFILE
    for directory in (profile, state_root / "workspace", home / "cache", home / "tmp"):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = profile / "package.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({
            "name": "tyloinfer-agent-profile", "private": True,
            "dependencies": {name: "file:" + (plugins_root / directory).as_posix() for name, directory in PRODUCTS.items()},
            "dsh": {"profile": {"bundles": BUNDLES}},
        }, indent=2) + "\n", encoding="utf-8")
    for name, directory in PRODUCTS.items():
        _link_directory(profile / "node_modules" / name, plugins_root / directory)
    patch = profile / "cordis.patch.yml"
    if not patch.exists():
        patch.write_text(json.dumps([
            {"id": "llm-pi-ai", "config": {"providers": {"tyloinfer": {
                "displayName": "TyloInfer", "api": "openai-completions", "baseURL": core_url,
                "models": [], "compat": {"maxTokensField": "max_tokens", "supportsDeveloperRole": False},
            }}}},
            {"id": "web-runtime", "config": {"openBrowser": False, "printUrl": False, "surfaceContext": True}},
            {"id": "session-log-deepseek", "config": {"enabled": False, "maxBytes": 8388608}},
            {"id": "plugin-package-inventory-deepseek", "config": {"enabled": False}},
            {"id": "tyloinfer-host", "config": {"provider": "tyloinfer", "connectionFile": str(state_root / "connection.json")}},
        ], indent=2) + "\n", encoding="utf-8")


def launch_command(carrier: Path, port: int) -> list[str]:
    """Use only the supported official profile entry point on a loopback listener."""
    if not 1 <= port <= 65535:
        raise LauncherError("The Agent port must be between 1 and 65535")
    return [str(carrier), "--profile", PROFILE, "--host", "127.0.0.1", "--port", str(port), "--no-open"]


def open_existing(state_root: Path, port: int, opener=None) -> None:
    """Open the private loopback connection in a local browser without printing it."""
    connection = _read_manifest(state_root / "connection.json", "existing Agent connection")
    raw = connection.get("url")
    declared = connection.get("origin")
    if not isinstance(raw, str) or not isinstance(declared, str) or any(char in raw for char in "\r\n\x00"):
        raise LauncherError("The existing Agent connection is invalid")
    try:
        url, origin = urlsplit(raw), urlsplit(declared)
        valid = (
            url.scheme in {"http", "https"} and origin.scheme == url.scheme
            and url.hostname == origin.hostname == "127.0.0.1"
            and url.port == origin.port == port and 1 <= port <= 65535
            and not url.username and not url.password and not origin.username and not origin.password
            and url.path in {"", "/", "/connect"} and not url.fragment
            and origin.path in {"", "/"} and not origin.query and not origin.fragment
        )
    except ValueError as error:
        raise LauncherError("The existing Agent connection is invalid") from error
    if not valid:
        raise LauncherError("The existing Agent connection must match the selected loopback port")
    try:
        opened = (opener or webbrowser.open)(raw, new=2)
    except (OSError, webbrowser.Error) as error:
        raise LauncherError("The local browser could not be opened") from error
    if not opened:
        raise LauncherError("The local browser did not accept the open request")


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=repo / "dsh")
    parser.add_argument("--plugins-root", type=Path, default=repo / "plugins")
    parser.add_argument("--state-root", type=Path, required=True, help="Private profile and workspace directory")
    parser.add_argument("--core-url", default="http://127.0.0.1:8090/v1", help="Initial endpoint for a new profile only")
    parser.add_argument("--port", type=int, default=8091)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check metadata and artifacts without writing or starting")
    mode.add_argument("--prepare-only", action="store_true", help="Prepare the private profile without starting services")
    mode.add_argument("--open-existing", action="store_true", help="Open the existing private local Host without starting another process")
    args = parser.parse_args(argv)
    try:
        runtime_root, plugins_root, state_root = (value.resolve() for value in (args.runtime_root, args.plugins_root, args.state_root))
        if args.open_existing:
            open_existing(state_root, args.port)
            print("AGENT_BROWSER_OPEN_REQUESTED; EXISTING_HOST_ONLY=true")
            return 0
        carrier = check_layout(runtime_root, plugins_root)
        check_profile_bindings(state_root, plugins_root)
        environment = isolated_environment(state_root, dict(os.environ))
        version = runtime_version(runtime_root)
        environment["TYLOINFER_DSH_VERSION"] = version
        command = launch_command(carrier, args.port)
        if args.check:
            print(json.dumps({"ready": True, "dsh": version, "bundles": BUNDLES, "starts_models": False}))
            return 0
        prepare_profile(state_root, plugins_root, args.core_url)
        if args.prepare_only:
            print("AGENT_PROFILE_PREPARED; CORE_FILES=unchanged; MODELS=none")
            return 0
        return subprocess.run(command, cwd=state_root / "workspace", env=environment, check=False).returncode
    except (LauncherError, OSError) as error:
        print(str(error) if isinstance(error, LauncherError) else "Cannot prepare or launch the isolated Agent Host", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
