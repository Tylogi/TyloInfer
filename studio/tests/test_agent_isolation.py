"""Inference API remains usable when all DSH Python imports are unavailable."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_core_api_starts_without_importing_dsh(tmp_path):
    root = Path(__file__).resolve().parents[2]
    program = r'''
import asyncio
import importlib.abc
import sys
from pathlib import Path

class RejectDsh(importlib.abc.MetaPathFinder):
    attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'deepseek_harness', 'deepseek_harness_runtime', 'tyloinfer_legacy_agent'} or fullname.startswith('server.agent'):
            self.attempts.append(fullname)
            raise ModuleNotFoundError('DSH unavailable in core isolation check')

blocker = RejectDsh()
sys.meta_path.insert(0, blocker)

import httpx
from server.api import create_app
from server.services.service import ServerService
from server.state.storage import SessionStore

class Backend:
    async def runtime_models(self):
        return {'object': 'list', 'data': []}

    async def aclose(self):
        pass

async def check():
    service = ServerService(SessionStore(Path(sys.argv[1]) / 'core.sqlite3'), Backend())
    app = create_app(service)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://core') as client:
            health = await client.get('/health')
            assert health.status_code == 200
            assert health.json()['service'] == 'tyloinfer-server'
            models = await client.get('/v1/models')
            assert models.status_code == 200 and models.json()['data'] == []
            assert not hasattr(app.state, 'agent')
            assert not any(path.startswith('/api/v1/agent') for path in app.openapi()['paths'])
            assert (await client.get('/api/v1/agent/capabilities')).status_code == 404
            assert (await client.get('/api/v1/agent/settings')).status_code == 404
    assert not blocker.attempts
    assert not any(name.split('.')[0] in {'deepseek_harness', 'deepseek_harness_runtime', 'tyloinfer_legacy_agent'} for name in sys.modules)
    print('CORE_WITHOUT_DSH_PASS')

asyncio.run(check())
'''
    environment = {**os.environ, "PYTHONPATH": str(root / "studio"), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(
        [sys.executable, "-B", "-c", program, str(tmp_path)],
        cwd=root,
        env=environment,
        check=True,
        timeout=30,
    )
