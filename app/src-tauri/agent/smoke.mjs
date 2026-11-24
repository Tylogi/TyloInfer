import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, join, delimiter } from 'node:path';
import { fileURLToPath } from 'node:url';

// Exercise the deployed graph, not the builder's imports or the user's profile.
const runtime = resolve(process.argv[2] ?? fileURLToPath(new URL('./runtime', import.meta.url)));
const home = mkdtempSync(join(tmpdir(), 'TyloInfer Agent 测试 '));
const started = performance.now();
const environment = { ...process.env, DSH_HOME: home, DSH_TELEMETRY_DISABLED: '1', NODE_OPTIONS: '', NODE_PATH: '' };
// Only OS utilities and bundled Node should be available, never build-host pnpm/Node.
environment.PATH = [runtime, ...(process.platform === 'win32' ? [join(process.env.SystemRoot, 'System32'), process.env.SystemRoot] : ['/usr/bin', '/bin'])].join(delimiter);
const child = spawn(join(runtime, process.platform === 'win32' ? 'node.exe' : 'node'), [join(runtime, 'boot.mjs'), 'launch'], {
  cwd: process.argv[3] ? resolve(process.argv[3]) : home, env: environment, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
});
let errors = '';
let pending = '';
let ready = false;
let failure;
let checking;
const redact = text => text.replace(/([?&]token=)[^\s&]+/g, '$1[redacted]');
// Match the desktop's readiness budget, rather than accepting a runtime that
// would time out when launched by the real application.
const timer = setTimeout(() => { failure = new Error('Agent did not become ready within 30 seconds.'); child.kill(); }, 30000);
child.stderr.on('data', chunk => { errors = (errors + chunk).slice(-32000); });
child.stdout.on('data', chunk => {
  pending += chunk;
  const lines = pending.split('\n');
  pending = lines.pop();
  for (const line of lines) {
    if (!line.startsWith('tyloinfer-agent ready: ') || ready) continue;
    ready = true;
    clearTimeout(timer);
    console.log(`Agent ready after ${((performance.now() - started) / 1000).toFixed(1)}s.`);
    checking = (async () => {
      const url = new URL(line.slice('tyloinfer-agent ready: '.length).trim());
      assert.equal(url.hostname, '127.0.0.1');
      assert.ok(url.searchParams.get('token'));
      const exchange = await fetch(url, { redirect: 'manual', signal: AbortSignal.timeout(10000) });
      assert.equal(exchange.status, 303);
      const cookie = exchange.headers.get('set-cookie')?.split(';')[0];
      assert.ok(cookie, 'Launch-token exchange did not set a browser cookie');
      const response = await fetch(new URL('/', url), { headers: { cookie }, signal: AbortSignal.timeout(10000) });
      assert.equal(response.status, 200);
      const html = await response.text();
      assert.ok(html.includes('__DSH_BOOT__'), 'DSH shell boot data missing');
      assert.ok(html.includes('@tyloinfer/dsh-plugin'), 'TyloInfer client plugin missing');
      console.log('Authenticated DSH shell and TyloInfer plugin verified.');
    })().catch(error => { failure = error; }).finally(() => {
      clearTimeout(timer);
      child.stdin.end();
      shutdownTimer = setTimeout(() => { failure ??= new Error('Agent did not stop after stdin EOF.'); child.kill(); }, 10000);
    });
  }
});
let shutdownTimer;
try {
  const code = await new Promise((resolve, reject) => { child.on('error', reject); child.on('close', resolve); });
  await checking;
  if (failure) throw failure;
  assert.ok(ready, `Agent exited before readiness (${code}).`);
  assert.equal(code, 0, 'Agent shutdown failed');
  console.log('Agent smoke test passed.');
} catch (error) {
  console.error(redact(`${error.stack}\n${errors}`));
  process.exitCode = 1;
} finally {
  clearTimeout(timer);
  clearTimeout(shutdownTimer);
  rmSync(home, { recursive: true, force: true });
}
