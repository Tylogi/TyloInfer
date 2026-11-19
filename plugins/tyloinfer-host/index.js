import { readFileSync, writeFileSync } from 'node:fs';
import z from '@deepseek-ai/schemastery';
import { createCoreBridge } from './core-bridge.js';

// Cordis 4.1 declares FiberState.ACTIVE as a compile-time enum, not a JS export.
const ACTIVE_STATE = 2;
const manifest = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8'));
const runtimeVersion = process.env.TYLOINFER_DSH_VERSION;
const dshVersion = typeof runtimeVersion === 'string' && /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/.test(runtimeVersion) ? runtimeVersion : null;

export const name = 'tyloinfer-host';
export const inject = ['settings', 'webServer', 'connection', 'loader'];
export const Config = z.object({
  provider: z.string().default('tyloinfer'),
  connectionFile: z.string(),
  requestTimeoutMs: z.natural().min(1).max(300000).default(5000),
  maintenanceTimeoutMs: z.natural().min(1).max(300000).default(30000),
});

export function apply(ctx, config) {
  const provider = () => ctx.settings.describe({ redactSecrets: true })
    .find(row => row.ns === 'llm-pi-ai')?.value?.providers?.[config.provider];
  const status = async () => {
    const selected = provider();
    let core = { connected: false, protocol_compatible: false, reason: 'provider_not_configured' };
    if (selected?.baseURL) {
      try {
        const base = new URL(selected.baseURL);
        if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password) throw new Error('endpoint');
        base.pathname = base.pathname.replace(/\/v1\/?$/, '') + '/health';
        base.search = base.hash = '';
        const headers = {};
        if (selected.apiKeyEnv) {
          const credential = await ctx.get('credentials')?.resolve(selected.apiKeyEnv);
          if (credential?.value) headers.Authorization = 'Bearer ' + credential.value;
        }
        const response = await fetch(base, { headers, signal: AbortSignal.timeout(3000), redirect: 'error' });
        const health = response.ok ? await response.json() : null;
        core = { connected: response.ok && health?.service === 'tyloinfer-server',
          protocol_compatible: health?.protocol_version === '1.0', reason: response.ok ? null : 'core_unavailable' };
      } catch { core = { connected: false, protocol_compatible: false, reason: 'core_unavailable' }; }
    }
    const entries = [...ctx.loader.entries()].map(entry => ({ id: entry.options.id, active: entry.fiber?.state === ACTIVE_STATE }));
    return { plugin: manifest.name, version: manifest.version, dsh_compatibility: manifest.engines.dsh, dsh_version: dshVersion,
      core, provider: config.provider, configured_models: selected?.models?.length ?? 0,
      settings_namespaces: ctx.settings.describe({ redactSecrets: true }).map(row => row.ns),
      entries };
  };
  ctx.effect(() => ctx.connection.rpc.handle('/tyloinfer', createCoreBridge({
    getProvider: provider,
    resolveCredential: reference => ctx.get('credentials')?.resolve(reference),
    timeoutMs: config.requestTimeoutMs,
    maintenanceTimeoutMs: config.maintenanceTimeoutMs,
  })));
  ctx.effect(() => ctx.webServer.register({ kind: 'exact', path: '/tyloinfer/status',
    async handler(req, res) {
      if (req.method !== 'GET') { res.writeHead(405); res.end(); return; }
      try {
        const value = await status();
        res.writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-store' });
        res.end(JSON.stringify(value));
      } catch { res.writeHead(503); res.end(); }
    }
  }));
  if (config.connectionFile) {
    let disposed = false;
    ctx.effect(() => () => { disposed = true; });
    void ctx.root.loader.await().then(() => {
      if (disposed) return;
      writeFileSync(config.connectionFile, JSON.stringify({
        url: ctx.connection.authenticatedUrl(ctx.webServer.getOrigin()),
        origin: ctx.webServer.getOrigin(), profile: 'tyloinfer-web'
      }), { mode: 0o600 });
    }).catch(error => ctx.logger.error('TyloInfer connection handoff failed: %s', error.name));
  }
}
