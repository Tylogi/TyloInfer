import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import z from '@deepseek-ai/schemastery';
import { createCoreBridge, readJson } from './core-bridge.js';

// Cordis 4.1 declares FiberState.ACTIVE as a compile-time enum, not a JS export.
const ACTIVE_STATE = 2;
const manifest = JSON.parse(readFileSync(new URL('../../package.json', import.meta.url), 'utf8'));
const runtimeVersion = process.env.TYLOINFER_DSH_VERSION;
const dshVersion = typeof runtimeVersion === 'string' && /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/.test(runtimeVersion) ? runtimeVersion : null;

// shortcut: ctx is the official Cordis context; type it when the bundle declares the Cordis package as a dependency.
interface HostConfig { provider: string; connectionFile: string; requestTimeoutMs: number; maintenanceTimeoutMs: number }

export const name = 'tyloinfer-host';
export const inject = ['settings', 'webServer', 'connection', 'loader'];
export const Config = z.object({
  provider: z.string().default('tyloinfer'),
  connectionFile: z.string().default(''),
  requestTimeoutMs: z.natural().min(1).max(300000).default(5000),
  maintenanceTimeoutMs: z.natural().min(1).max(300000).default(30000),
});

export function apply(ctx: any, config: HostConfig) {
  const providerSettings = () => ctx.settings.describe({ redactSecrets: true }).find((row: any) => row.ns === 'llm-pi-ai');
  const provider = () => providerSettings()?.value?.providers?.[config.provider];
  ctx.effect(() => ctx.on('llm-pi-ai/request-headers', (request: any, _headers: any, next: any) => {
    const headers = next();
    if (request.provider !== config.provider || headers.get('authorization')?.trim() ||
        headers.get('cf-aig-authorization')?.trim()) return headers;
    const selected = provider();
    if (!selected?.baseURL || selected.api !== 'openai-completions' || selected.apiKeyEnv) return headers;
    try {
      const target = new URL(selected.baseURL);
      const endpoint = new URL(request.baseURL);
      if (!['http:', 'https:'].includes(target.protocol) ||
          !['127.0.0.1', '[::1]', 'localhost'].includes(target.hostname) ||
          target.username || target.password || target.search || target.hash ||
          target.href.replace(/\/$/, '') !== endpoint.href.replace(/\/$/, '')) return headers;
    } catch { return headers; }
    // pi-ai requires an authentication header even for an unauthenticated OpenAI-compatible server.
    headers.set('authorization', 'TyloInfer-Local');
    return headers;
  }));
  const matchesProvider = (left: any, right: any) => left?.baseURL === right.baseURL && left?.apiKeyEnv === right.apiKeyEnv;
  const lifetime = new AbortController();
  let discoveryTimer: ReturnType<typeof setTimeout> | undefined;
  let discoveryRunning = false;
  let discoveryRequested = false;
  async function syncProviderModels() {
    const selected = provider();
    if (!selected?.baseURL || !ctx.settings.writable) return;
    const root = new URL(selected.baseURL);
    if (!['http:', 'https:'].includes(root.protocol) || root.username || root.password || root.search || root.hash) return;
    const prefix = root.pathname.replace(/\/v1\/?$/, '').replace(/\/$/, '');
    const credential = selected.apiKeyEnv ? (await ctx.get('credentials')?.resolve(selected.apiKeyEnv))?.value : '';
    if (selected.apiKeyEnv && !credential) return;
    const signal = AbortSignal.any([lifetime.signal, AbortSignal.timeout(config.maintenanceTimeoutMs)]);
    async function read(route: string) {
      const url = new URL(root);
      url.pathname = prefix + route;
      const headers = credential ? { Authorization: 'Bearer ' + credential } : {};
      const response = await fetch(url, { headers, signal, redirect: 'error' });
      if (!response.ok) throw new Error('TyloInfer discovery is unavailable');
      return readJson(response, credential, 4 * 1024 * 1024);
    }
    const health = await read('/health');
    if (health?.service !== 'tyloinfer-server' || health.protocol_version !== '1.0') return;
    const [listing, mappings] = await Promise.all([
      read('/api/v1/runtime/models'), read('/api/v1/runtime/model-aliases'),
    ]);
    const validId = (id: any) => typeof id === 'string' && id.trim() && !/[\x00-\x1f\x7f]/.test(id);
    if (!Array.isArray(listing?.data) || listing.data.some((model: any) => !validId(model?.id))) return;
    const aliases = mappings?.aliases ?? {};
    const ids = [...new Set(listing.data.map(({ id }: { id: string }) => Object.hasOwn(aliases, id) && validId(aliases[id]) ? aliases[id] : id))];
    const current = providerSettings();
    const currentProvider = current?.value?.providers?.[config.provider];
    if (lifetime.signal.aborted || !matchesProvider(currentProvider, selected)) return;
    const previous = Array.isArray(currentProvider.models) ? currentProvider.models : [];
    if (ids.length !== previous.length || ids.some((id, index) => id !== previous[index]?.id)) {
      const existing = new Map(previous.map((model: any) => [model.id, model]));
      await ctx.settings.mutate(current.ns, [{ op: 'set', path: ['providers', config.provider, 'models'],
        value: ids.map(id => existing.get(id) ?? { id }) }], current.revision);
    }
    const defaultModel = ctx.get('agentDefaultModel');
    const selection = defaultModel?.currentSelection();
    if (!lifetime.signal.aborted && ids.length && selection?.provider === config.provider && !selection.model) {
      await defaultModel.saveSelection({ provider: config.provider, model: ids[0] });
    }
  }
  function scheduleDiscovery(delay = 0) {
    if (lifetime.signal.aborted) return;
    clearTimeout(discoveryTimer);
    discoveryTimer = setTimeout(async () => {
      if (discoveryRunning) { discoveryRequested = true; return; }
      discoveryRunning = true;
      let nextDelay = 5000;
      try { await syncProviderModels(); }
      catch { nextDelay = 15000; }
      finally {
        discoveryRunning = false;
        const delay = discoveryRequested ? 0 : nextDelay;
        discoveryRequested = false;
        scheduleDiscovery(delay);
      }
    }, delay);
  }
  ctx.effect(() => () => { lifetime.abort(); clearTimeout(discoveryTimer); });
  void ctx.root.loader.await().then(() => scheduleDiscovery());
  async function prepareProviderUpdate(previous: any, { baseURL, apiKey }: { baseURL?: string; apiKey?: string }) {
    if (!ctx.settings.writable || !matchesProvider(provider(), previous)) throw new Error('Core provider configuration cannot be updated');
    const patch: Record<string, any> = {};
    if (baseURL) patch.baseURL = baseURL;
    const credentials = ctx.get('credentials');
    if (apiKey) {
      const digest = createHash('sha256').update(config.provider).digest('hex').slice(0, 16).toUpperCase();
      const prefix = 'TYLOINFER_CORE_KEY_' + digest;
      patch.apiKeyEnv = previous.apiKeyEnv === prefix + '_A' ? prefix + '_B' : prefix + '_A';
      await credentials.set(patch.apiKeyEnv, apiKey);
    }
    return {
      async commit() {
        const current = providerSettings();
        if (!matchesProvider(current?.value?.providers?.[config.provider], previous)) throw new Error('Core provider configuration changed');
        await ctx.settings.mutate(current.ns, Object.entries(patch).map(([field, value]) => ({
          op: 'set', path: ['providers', config.provider, field], value,
        })), current.revision);
      },
      async discard() {
        if (patch.apiKeyEnv && provider()?.apiKeyEnv !== patch.apiKeyEnv) await credentials.unset(patch.apiKeyEnv);
      },
    };
  }
  const status = async () => {
    const selected = provider();
    let core: { connected: boolean; protocol_compatible: boolean; reason: string | null } = { connected: false, protocol_compatible: false, reason: 'provider_not_configured' };
    if (selected?.baseURL) {
      try {
        const base = new URL(selected.baseURL);
        if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password) throw new Error('endpoint');
        base.pathname = base.pathname.replace(/\/v1\/?$/, '') + '/health';
        base.search = base.hash = '';
        const headers: Record<string, string> = {};
        if (selected.apiKeyEnv) {
          const credential = await ctx.get('credentials')?.resolve(selected.apiKeyEnv);
          if (credential?.value) headers.Authorization = 'Bearer ' + credential.value;
        }
        const response = await fetch(base, { headers, signal: AbortSignal.timeout(3000), redirect: 'error' });
        const health: any = response.ok ? await response.json() : null;
        core = { connected: response.ok && health?.service === 'tyloinfer-server',
          protocol_compatible: health?.protocol_version === '1.0', reason: response.ok ? null : 'core_unavailable' };
      } catch { core = { connected: false, protocol_compatible: false, reason: 'core_unavailable' }; }
    }
    const entries = [...ctx.loader.entries()].map((entry: any) => ({ id: entry.options.id, active: entry.fiber?.state === ACTIVE_STATE }));
    return { plugin: manifest.name, version: manifest.version, dsh_compatibility: manifest.engines.dsh, dsh_version: dshVersion,
      core, provider: config.provider, configured_models: selected?.models?.length ?? 0,
      settings_namespaces: ctx.settings.describe({ redactSecrets: true }).map((row: any) => row.ns),
      entries };
  };
  const bridge = createCoreBridge({
    getProvider: provider,
    resolveCredential: (reference: string) => ctx.get('credentials')?.resolve(reference),
    prepareProviderUpdate,
    timeoutMs: config.requestTimeoutMs,
    maintenanceTimeoutMs: config.maintenanceTimeoutMs,
  });
  ctx.effect(() => ctx.connection.rpc.handle('/tyloinfer', async (endpoint: string, payload: any, signal: AbortSignal, peer: any) => {
    const result = await bridge(endpoint, payload, signal, peer);
    if (!result.ok) return result;
    if (endpoint === 'service/configure' && payload?.resource === 'model_aliases' ||
        endpoint === 'service/configure-listener' || endpoint === 'service/configure-authentication') {
      scheduleDiscovery();
    }
    return result;
  }));
  ctx.effect(() => ctx.webServer.register({ kind: 'exact', path: '/tyloinfer/status',
    async handler(req: any, res: any) {
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
    }).catch((error: any) => ctx.logger.error('TyloInfer connection handoff failed: %s', error.name));
  }
}
