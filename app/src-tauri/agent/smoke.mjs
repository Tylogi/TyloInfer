import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { delimiter, join, resolve, sep } from 'node:path';
import { createInterface } from 'node:readline';

const runtime = resolve(process.argv[2] ?? '');
assert.ok(process.argv[2], 'Usage: node smoke.mjs path/to/agent-runtime');
const temporary = mkdtempSync(join(tmpdir(), 'tylo-agent-smoke-'));
const workspace = join(temporary, 'workspace');
mkdirSync(workspace);
const windows = process.platform === 'win32';
const environment = {
  DSH_HOME: join(temporary, 'profile'),
  PATH: [runtime, ...(windows ? [join(process.env.SystemRoot ?? 'C:\\Windows', 'System32')] : ['/usr/bin', '/bin'])].join(delimiter),
  TEMP: temporary, TMP: temporary, TMPDIR: temporary,
};
for (const name of ['SystemRoot', 'WINDIR', 'ComSpec', 'SYSTEMDRIVE']) {
  if (process.env[name]) environment[name] = process.env[name];
}
const child = spawn(join(runtime, windows ? 'node.exe' : 'node'), [join(runtime, 'boot.mjs'), 'launch'], {
  cwd: workspace, env: environment, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true,
});
let diagnostics = '';
child.stderr.on('data', chunk => { diagnostics = (diagnostics + chunk).slice(-8192); });

async function stop() {
  if (child.exitCode !== null || child.signalCode !== null) return;
  let timer;
  const exited = new Promise(resolve => child.once('exit', resolve));
  child.stdin.end();
  const clean = await Promise.race([
    exited.then(() => true),
    new Promise(resolve => { timer = setTimeout(() => resolve(false), 6000); }),
  ]);
  clearTimeout(timer);
  if (!clean) {
    child.kill();
    await exited;
    throw new Error('Agent did not shut down after stdin closed');
  }
  assert.equal(child.exitCode, 0, 'Agent shutdown failed');
}

try {
  let timer;
  const reader = createInterface({ input: child.stdout });
  const ready = await new Promise((resolve, reject) => {
    timer = setTimeout(() => reject(new Error('Agent readiness timed out after 30 seconds')), 30000);
    child.once('error', reject);
    child.once('exit', code => reject(new Error('Agent exited before readiness: ' + code)));
    reader.on('line', line => {
      const prefix = 'tyloinfer-agent ready: ';
      if (line.startsWith(prefix)) {
        try { resolve(JSON.parse(line.slice(prefix.length))); } catch { reject(new Error('Invalid Agent readiness message')); }
      }
    });
  }).finally(() => clearTimeout(timer));
  let launch;
  try { launch = new URL(ready.url); } catch { throw new Error('Invalid Agent launch URL'); }
  assert.equal(launch.protocol, 'http:');
  assert.equal(launch.hostname, '127.0.0.1');
  assert.ok(launch.port && launch.searchParams.get('token'), 'Agent launch authentication is missing');
  const cookies = new Map();
  let url = launch;
  let response;
  for (let redirects = 0; redirects < 6; redirects++) {
    assert.equal(url.origin, launch.origin, 'Agent redirected outside its local origin');
    response = await fetch(url, { redirect: 'manual', signal: AbortSignal.timeout(10000),
      headers: { cookie: [...cookies].map(([name, value]) => name + '=' + value).join('; ') } });
    for (const cookie of response.headers.getSetCookie()) {
      const pair = cookie.split(';')[0], index = pair.indexOf('=');
      if (index > 0) cookies.set(pair.slice(0, index), pair.slice(index + 1));
    }
    if (response.status < 300 || response.status >= 400) break;
    const location = response.headers.get('location');
    assert.ok(location, 'Agent authentication redirect has no destination');
    url = new URL(location, url);
    await response.arrayBuffer();
  }
  assert.equal(response.status, 200, 'Authenticated Agent shell did not load');
  const html = await response.text();
  assert.match(response.headers.get('content-type') ?? '', /text\/html/i);
  // Match complete tags so script-like strings inside the inline loader are not assets.
  const tags = [...html.matchAll(/<script\b([^>]*)>[\s\S]*?<\/script\s*>/gi)];
  assert.ok(tags.length, 'Agent shell has no JavaScript entry point');
  const scripts = tags.flatMap(([, attributes]) => {
    const source = /(?:^|\s)src=["']([^"']+)["']/i.exec(attributes);
    return source ? [source[1]] : [];
  });
  for (const source of scripts) {
    const asset = new URL(source.replaceAll('&amp;', '&'), url);
    assert.equal(asset.origin, launch.origin, 'Agent script has an external origin');
    const loaded = await fetch(asset, { signal: AbortSignal.timeout(10000),
      headers: { cookie: [...cookies].map(([name, value]) => name + '=' + value).join('; ') } });
    assert.equal(loaded.status, 200, 'Agent JavaScript asset did not load: ' + asset.pathname);
    assert.ok((await loaded.arrayBuffer()).byteLength, 'Agent JavaScript asset is empty');
  }
  await stop();
  console.log('AGENT_SMOKE_OK authenticated_shell=true scripts=' + scripts.length + ' graceful_shutdown=true');
} catch (error) {
  console.error(error.message);
  // Never include the short-lived launch token in build logs.
  console.error(diagnostics.replace(/([?&]token=)[^&\s"'<>]+/g, '$1[redacted]'));
  process.exitCode = 1;
} finally {
  await stop();
  assert.ok(resolve(temporary).startsWith(resolve(tmpdir()) + sep));
  rmSync(temporary, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
}
