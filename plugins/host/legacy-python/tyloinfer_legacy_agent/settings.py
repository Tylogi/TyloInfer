"""Bounded transport to official DSH settings in a server-owned private profile."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import httpx

from server.server.services.service import ServiceError

from .dsh import DshRuntime

PROFILE = "tyloinfer-settings"
EDITABLE = {
    "ui-theme": ["preference", "fontSize", "codeFontSize", "textFontFamily", "codeFontFamily"],
    "locale": ["preference"],
    "ui-chat": [],
    "ui-conversation": [],
    "ui-settings": [],
}

SETTINGS_PLUGIN = r"""
import http from 'node:http';
import { timingSafeEqual } from 'node:crypto';
export const name = 'tyloinfer-settings-transport';
export const inject = ['settings'];
const fields = {
  'ui-theme': ['preference', 'fontSize', 'codeFontSize', 'textFontFamily', 'codeFontFamily'],
  'locale': ['preference'], 'ui-chat': [], 'ui-conversation': [], 'ui-settings': []
};
export function apply(ctx) {
  let serial = Promise.resolve();
  const describe = () => ({ writable: true, hasDocument: false,
    namespaces: ctx.settings.describe({redactSecrets: true})
      .filter(row => Object.hasOwn(fields, row.ns))
      .map(row => ({...row, editable_fields: fields[row.ns]})) });
  const server = http.createServer(async (req, res) => {
    const answer = (status, value) => {
      res.writeHead(status, {'content-type': 'application/json', 'cache-control': 'no-store'});
      res.end(JSON.stringify(value));
    };
    const supplied = Buffer.from(req.headers.authorization ?? '');
    const expected = Buffer.from('Bearer ' + process.env.TYLOINFER_SETTINGS_TOKEN);
    if (supplied.length !== expected.length || !timingSafeEqual(supplied, expected)) {
      answer(403, {code: 'settings_forbidden'}); return;
    }
    if (req.method !== 'POST' || req.url !== '/settings') {
      answer(404, {code: 'settings_not_found'}); return;
    }
    try {
      const chunks = [];
      let bytes = 0;
      for await (const chunk of req) {
        bytes += chunk.length;
        if (bytes > 65536) throw new Error('size');
        chunks.push(chunk);
      }
      const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      if (!body || typeof body !== 'object' || Array.isArray(body)) throw new Error('rejected');
      const operation = async () => {
        if (body.mode === 'describe') return describe();
        const allowed = Object.hasOwn(fields, body.ns) ? fields[body.ns] : undefined;
        if (!allowed?.length || !['update', 'replace'].includes(body.mode)
          || !Number.isSafeInteger(body.expected_revision) || body.expected_revision < 0
          || !body.values || typeof body.values !== 'object' || Array.isArray(body.values)
          || Object.keys(body.values).some(key => !allowed.includes(key))) throw new Error('rejected');
        await ctx.settings[body.mode](body.ns, body.values, body.expected_revision);
        const value = describe().namespaces.find(row => row.ns === body.ns);
        if (!value) throw new Error('unavailable');
        return value;
      };
      const result = serial.then(operation);
      serial = result.catch(() => {});
      answer(200, await result);
    } catch (error) {
      answer(error?.code === 'SETTINGS_CONFLICT' ? 409 : 422,
        {code: error?.code === 'SETTINGS_CONFLICT' ? 'settings_conflict' : 'settings_rejected'});
    }
  });
  server.requestTimeout = 15000;
  server.headersTimeout = 10000;
  ctx.effect(() => {
    server.listen(0, '127.0.0.1', () => {
      fetch(process.env.TYLOINFER_SETTINGS_READY, {method: 'POST',
        headers: {'content-type': 'application/json', 'authorization': 'Bearer ' + process.env.TYLOINFER_SETTINGS_TOKEN},
        body: JSON.stringify({port: server.address().port})
      }).catch(() => server.close());
    });
    return () => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); });
  });
}
"""


class NativeSettings:
    def __init__(self, root: Path):
        self.root = root
        self.harness = None
        self.client = None
        self.generation = str(uuid4())
        self.lock = asyncio.Lock()
        self.closed = False

    def _prepare(self):
        if os.environ.get("DSH_OFFICE_SIDECAR"):
            raise RuntimeError("Inherited Office resources are not permitted")
        home, workspace = self.root / "home", self.root / "workspace"
        profile = home / "profiles" / PROFILE
        for directory in (profile, workspace, home / "cache", home / "tmp"):
            directory.mkdir(parents=True, exist_ok=True)
        manifest = profile / "package.json"
        if not manifest.exists():
            manifest.write_text(
                json.dumps(
                    {
                        "private": True,
                        "dependencies": {},
                        "dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-sdk-minimal"]}},
                    }
                ),
                encoding="utf-8",
            )
        plugin = self.root / "settings-transport.mjs"
        plugin.write_text(SETTINGS_PLUGIN, encoding="utf-8")
        patch = profile / "cordis.patch.yml"
        if not patch.exists():
            disabled = [
                "llm-deepseek",
                "deepseek-llm-api-extensions",
                "session-log-deepseek",
                "plugin-package-inventory-deepseek",
                "persistent-bash",
                "persistent-pwsh",
                "terminal-bash",
                "terminal-pwsh",
                "tool-fs",
                "tool-working-directory",
                "compaction-basic",
                "image-offload",
            ]
            entries = [
                {"id": "config-editor", "name": "@deepseek-ai/dsh-config-editor"},
                {"id": "settings", "name": "@deepseek-ai/dsh-settings"},
                {"id": "ui-theme", "name": "@deepseek-ai/dsh-client-ui-theme"},
                {"id": "locale", "name": "@deepseek-ai/dsh-client-locale"},
                {"id": "ui-chat", "name": "@deepseek-ai/dsh-client-ui-chat"},
                {"id": "ui-conversation", "name": "@deepseek-ai/dsh-client-ui-conversation"},
                {"id": "ui-settings", "name": "@deepseek-ai/dsh-client-ui-settings"},
                {"id": "tyloinfer-settings-transport", "name": str(plugin.resolve())},
            ]
            patch.write_text(
                json.dumps(
                    [*({"id": key, "disabled": True} for key in disabled), {"insert": entries}]
                ),
                encoding="utf-8",
            )
        return home, workspace

    async def _ensure(self):
        if self.closed:
            raise ServiceError(503, "settings_unavailable", "Settings service is stopping")
        if self.client is not None:
            return
        if not DshRuntime.capabilities()["available"]:
            raise ServiceError(503, "settings_unavailable", "Install the pinned Agent runtime")
        from deepseek_harness import DeepSeekHarness

        home, workspace = self._prepare()
        token = secrets.token_urlsafe(32)
        ready = threading.Event()
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                self.connection.settimeout(5)
                try:
                    if not secrets.compare_digest(
                        self.headers.get("Authorization", ""), "Bearer " + token
                    ):
                        self.send_error(403)
                        return
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 1024:
                        raise ValueError("Invalid readiness request")
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError("Invalid readiness response")
                    port = body.get("port")
                    if type(port) is not int or not 1 <= port <= 65535:
                        raise ValueError("Invalid readiness port")
                    received["port"] = port
                    self.send_response(204)
                    self.end_headers()
                    ready.set()
                except (ValueError, OSError):
                    self.send_error(400)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env = {
            key: ""
            for key in os.environ
            if any(part in key.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "KEY"))
            or (key.startswith("DSH_") and key != "DSH_OFFICE_SIDECAR")
        }
        env.update(
            {
                "TYLOINFER_SETTINGS_TOKEN": token,
                "TYLOINFER_SETTINGS_READY": f"http://127.0.0.1:{server.server_port}/ready",
                "HOME": str(workspace),
                "USERPROFILE": str(workspace),
                "APPDATA": str(home / "cache"),
                "LOCALAPPDATA": str(home / "cache"),
                "XDG_CONFIG_HOME": str(home / "cache"),
                "DSH_RESOURCE_CACHE": str(home / "cache"),
                "TEMP": str(home / "tmp"),
                "TMP": str(home / "tmp"),
                "HTTP_PROXY": "",
                "HTTPS_PROXY": "",
                "ALL_PROXY": "",
                "NO_PROXY": "127.0.0.1,localhost",
                "NODE_OPTIONS": "",
                "NODE_PATH": "",
                "DSH_TELEMETRY_DISABLED": "1",
            }
        )
        self.harness = DeepSeekHarness(
            dsh_bin=os.environ.get("TYLOINFER_AGENT_DSH_BIN"),
            cwd=str(workspace),
            runtime_cwd=str(workspace),
            dsh_home=str(home),
            profile=PROFILE,
            env=env,
            initialize_timeout_seconds=30,
            shutdown_timeout_seconds=3,
        )
        try:
            await asyncio.to_thread(self.harness.client.start)
            if not await asyncio.to_thread(ready.wait, 30):
                raise RuntimeError("Settings transport did not start")
            self.client = httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{received['port']}",
                trust_env=False,
                headers={"Authorization": "Bearer " + token},
                timeout=30,
            )
            self.generation = str(uuid4())
        except BaseException:
            await asyncio.to_thread(self.harness.close)
            self.harness = None
            raise
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            await asyncio.to_thread(thread.join, 2)

    async def _request(self, body):
        try:
            await self._ensure()
            response = await self.client.post("/settings", json=body)
        except asyncio.CancelledError:
            raise
        except ServiceError:
            raise
        except Exception as error:
            await self._stop()
            raise ServiceError(
                503, "settings_unavailable", "The isolated DSH settings service is unavailable"
            ) from error
        if response.status_code == 409:
            raise ServiceError(409, "settings_conflict", "Settings changed; reload before saving")
        if response.status_code != 200:
            raise ServiceError(422, "settings_rejected", "DSH rejected the supplied settings")
        return response.json()

    async def describe(self):
        async with self.lock:
            value = await self._request({"mode": "describe"})
            return {
                **value,
                "generation": self.generation,
                "runtime": "dsh",
                "version": "0.2.2-rc.2",
                "persistence": "isolated-profile",
                "unavailable_sections": ["account", "models", "agent-presets", "plugins"],
            }

    async def update(self, namespace, *, mode, values, expected_revision, generation):
        async with self.lock:
            if namespace not in EDITABLE or not EDITABLE[namespace]:
                raise ServiceError(
                    403, "settings_read_only", "This settings consumer is not connected"
                )
            if set(values) - set(EDITABLE[namespace]):
                raise ServiceError(
                    422, "settings_rejected", "This field has no connected settings consumer"
                )
            await self._ensure()
            if generation != self.generation:
                raise ServiceError(
                    409, "settings_conflict", "Settings service restarted; reload before saving"
                )
            result = await self._request(
                {
                    "mode": mode,
                    "ns": namespace,
                    "values": values,
                    "expected_revision": expected_revision,
                }
            )
            return {**result, "generation": self.generation}

    async def _stop(self):
        client, self.client = self.client, None
        harness, self.harness = self.harness, None
        self.generation = str(uuid4())
        try:
            if client is not None:
                await client.aclose()
        finally:
            if harness is not None:
                await asyncio.to_thread(harness.close)

    async def aclose(self):
        async with self.lock:
            self.closed = True
            await self._stop()
