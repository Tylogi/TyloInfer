"""Shared preconditions for the Python to plugin Host cross-process tests."""

import shutil
import subprocess
from pathlib import Path

import pytest

# app/plugins/package.json engines.node; the smoke entries run as TypeScript.
MINIMUM_NODE = (24, 9)


def require_built_plugin_host() -> str:
    """Return Node.js that can import the built plugin Host module."""
    node = shutil.which('node')
    if node is None:
        pytest.skip('requires Node.js for the product Host module')
    version = subprocess.run([node, '--version'], capture_output=True, text=True, check=True, timeout=60).stdout
    release = tuple(int(part) for part in version.strip().lstrip('v').split('.')[:2])
    if release < MINIMUM_NODE:
        pytest.skip(f'the plugin Host module requires Node.js >= {MINIMUM_NODE[0]}.{MINIMUM_NODE[1]}, found {version.strip()}')
    root = Path(__file__).resolve().parents[3]
    if (root / 'app/plugins/lib/host/core-bridge.js').is_file():
        return node
    if shutil.which('pnpm') is None:
        pytest.skip('the plugin Host module is not built and pnpm is unavailable')
    subprocess.run(['pnpm', '--dir', str(root / 'app'), 'run', 'build:plugins'], check=True, timeout=900)
    return node
