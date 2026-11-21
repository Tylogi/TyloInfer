import { resolveCoreRead, resolveModelAnalysis, resolveModelDetails } from './page-reads.js';
import { resolvePageAction } from './page-actions.js';
import { resolveMaintenanceCall } from './maintenance-routes.js';
import { resolveModelAction } from './model-actions.js';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const MODEL_JOBS = new Set(['model.load', 'model.unload']);
const JOB_STATES = new Set(['queued', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled', 'interrupted']);
const CANCELLABLE = new Set(['queued', 'running']);
const MAX_REQUEST_BYTES = 65536;
const MAX_RESPONSE_BYTES = 4194304;
const MAX_ANALYSIS_RESPONSE_BYTES = 32 * 1024 * 1024;
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const exact = (value, key) => record(value) && Object.keys(value).length === 1 && Object.hasOwn(value, key);
const uuid = value => typeof value === 'string' && UUID.test(value);
const companionPath = value => typeof value === 'string' && value.length <= 4096 &&
  /^(?:\/(?!\/)|[A-Za-z]:[\\/])/.test(value) && !/[\x00-\x1f\x7f]/.test(value);

class BridgeFailure extends Error {
  constructor(code, message, details = {}) { super(message); this.code = code; this.details = details; }
}
const badRequest = () => new BridgeFailure('tyloinfer/bad-request', 'Choose a supported core operation and request');
const unknownOutcome = operationId => new BridgeFailure('tyloinfer/outcome-unknown',
  'The write may have been accepted; inspect current state before any manual retry',
  { operation_id: operationId, may_have_been_applied: true, retryable: false });

function parseCall(endpoint, payload) {
  if (endpoint === 'core/read' || endpoint === 'models/analysis') {
    const call = endpoint === 'core/read' ? resolveCoreRead(payload) : resolveModelAnalysis(payload);
    if (!call) throw badRequest();
    return call;
  }
  if (endpoint === 'models/cache-profile' || endpoint === 'models/directory' || endpoint === 'models/load-plan') {
    const call = resolveModelDetails(endpoint, payload);
    if (!call) throw badRequest();
    return call;
  }
  if (endpoint === 'models/load' || endpoint === 'models/unload') {
    const loadFields = endpoint === 'models/load' ? ['request', 'companion_ids', 'companion_paths'] : ['request'];
    if (!record(payload) || !Object.hasOwn(payload, 'request') || !Object.keys(payload).every(key => loadFields.includes(key)) || !record(payload.request)) throw badRequest();
    if (endpoint === 'models/load' && (typeof payload.request.model !== 'string' || !payload.request.model.trim()
      || Object.hasOwn(payload.request, 'gguf_companions'))) throw badRequest();
    if (Object.hasOwn(payload, 'companion_ids') && (!Array.isArray(payload.companion_ids) || payload.companion_ids.length > 16
      || !Array.from(payload.companion_ids).every(value => typeof value === 'string' && /^[0-9a-f]{32}$/.test(value))
      || new Set(payload.companion_ids).size !== payload.companion_ids.length)) throw badRequest();
    if (Object.hasOwn(payload, 'companion_paths') && (Object.hasOwn(payload, 'companion_ids')
      || !Array.isArray(payload.companion_paths) || payload.companion_paths.length > 16
      || !Array.from(payload.companion_paths).every(companionPath)
      || new Set(payload.companion_paths).size !== payload.companion_paths.length)) throw badRequest();
    if (endpoint === 'models/unload' && (!uuid(payload.request.instance_id)
      || (Object.hasOwn(payload.request, 'force') && typeof payload.request.force !== 'boolean'))) throw badRequest();
    let body;
    try {
      const request = Object.hasOwn(payload, 'companion_paths')
        ? { ...payload.request, gguf_companions: payload.companion_paths } : payload.request;
      body = JSON.stringify(request, (_key, value) => {
        if (value === undefined || typeof value === 'function' || typeof value === 'symbol'
          || typeof value === 'bigint' || (typeof value === 'number' && !Number.isFinite(value))) throw badRequest();
        return value;
      });
    } catch { throw badRequest(); }
    if (Buffer.byteLength(body) > MAX_REQUEST_BYTES) throw badRequest();
    return { kind: 'write', path: endpoint === 'models/load' ? '/api/v1/models/load' : '/api/v1/models/unload', body, method: 'POST', format: 'accepted',
      ...(Object.hasOwn(payload, 'companion_ids') ? { companionIds: payload.companion_ids } : {}) };
  }
  if (['operations/get', 'operations/cancel'].includes(endpoint) && exact(payload, 'operation_id') && uuid(payload.operation_id)) {
    return { kind: endpoint === 'operations/get' ? 'operation' : 'cancel', id: payload.operation_id, jobKinds: MODEL_JOBS };
  }
  const pageCall = resolvePageAction(endpoint, payload);
  if (pageCall) return pageCall;
  const maintenanceCall = resolveMaintenanceCall(endpoint, payload);
  if (maintenanceCall) return maintenanceCall;
  const modelAction = resolveModelAction(endpoint, payload);
  if (modelAction) return modelAction;
  throw badRequest();
}

export async function readJson(response, credential, maxBytes) {
  if (!response.body) throw new BridgeFailure('tyloinfer/core-invalid-response', 'The core returned no JSON body');
  const reader = response.body.getReader();
  const chunks = [];
  let length = 0;
  try {
    while (true) {
      const item = await reader.read();
      if (item.done) break;
      length += item.value.byteLength;
      if (length > maxBytes) {
        await reader.cancel();
        throw new BridgeFailure('tyloinfer/core-invalid-response', 'The core response exceeded the transport limit');
      }
      chunks.push(item.value);
    }
  } finally { reader.releaseLock(); }
  const text = Buffer.concat(chunks).toString('utf8');
  let value;
  try { value = JSON.parse(text); }
  catch { throw new BridgeFailure('tyloinfer/core-invalid-response', 'The core returned invalid JSON'); }
  if (credential) {
    const pending = [value];
    while (pending.length) {
      const item = pending.pop();
      if (typeof item === 'string' && item.includes(credential)) throw new BridgeFailure('tyloinfer/core-invalid-response', 'The core response contained protected credential material');
      if (record(item) || Array.isArray(item)) {
        for (const key of Object.keys(item)) {
          if (key.includes(credential)) throw new BridgeFailure('tyloinfer/core-invalid-response', 'The core response contained protected credential material');
          pending.push(item[key]);
        }
      }
    }
  }
  return value;
}

function untilAborted(promise, signal) {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const abort = () => reject(signal.reason);
    signal.addEventListener('abort', abort, { once: true });
    Promise.resolve(promise).then(value => {
      signal.removeEventListener('abort', abort);
      resolve(value);
    }, error => {
      signal.removeEventListener('abort', abort);
      reject(error);
    });
  });
}

function managedJob(value, id, jobKinds) {
  if (!record(value) || !uuid(value.id) || (id !== null && value.id.toLowerCase() !== id.toLowerCase())
    || !jobKinds.has(value.kind) || !JOB_STATES.has(value.status) || typeof value.cancel_requested !== 'boolean') {
    throw new BridgeFailure('tyloinfer/operation-not-supported', 'Only native operations supported by this endpoint are exposed');
  }
  return value;
}

export function createCoreBridge({ getProvider, resolveCredential, prepareProviderUpdate, fetchImpl = globalThis.fetch,
  timeoutMs = 5000, maintenanceTimeoutMs = 30000, maxResponseBytes }) {
  let changingConnection = false;
  return async (endpoint, payload, callerSignal, peer) => {
    const failure = error => ({ ok: false, error: { code: error.code, message: error.message, details: error.details } });
    if (peer?.principal?.kind !== 'owner') return failure(new BridgeFailure('tyloinfer/forbidden', 'Core access requires the owner'));
    let submitted = false;
    let operationId = null;
    let connectionChange = false;
    let providerUpdate;
    try {
      const call = parseCall(endpoint, payload);
      if (call.authChange || call.listenerChange) {
        if (changingConnection) throw new BridgeFailure('tyloinfer/connection-busy', 'A core connection change is already in progress');
        connectionChange = changingConnection = true;
      }
      operationId = call.id ?? null;
      const selected = getProvider();
      if (!selected?.baseURL) throw new BridgeFailure('tyloinfer/core-unavailable', 'TyloInfer provider is not configured');
      const base = new URL(selected.baseURL);
      if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash) {
        throw new BridgeFailure('tyloinfer/core-incompatible', 'The core endpoint is not a supported credential-free HTTP URL');
      }
      const prefix = base.pathname.replace(/\/v1\/?$/, '').replace(/\/$/, '');
      const signal = AbortSignal.any([callerSignal ?? new AbortController().signal, AbortSignal.timeout(call.maintenance ? maintenanceTimeoutMs : timeoutMs)]);
      signal.throwIfAborted();
      const headers = {};
      let credential = '';
      if (selected.apiKeyEnv) {
        const resolved = await untilAborted(resolveCredential(selected.apiKeyEnv), signal);
        if (typeof resolved?.value !== 'string' || !resolved.value) throw new BridgeFailure('tyloinfer/core-unavailable', 'The configured core credential is unavailable');
        credential = resolved.value;
        headers.Authorization = 'Bearer ' + credential;
      }
      const request = async (path, method = 'GET', body, query = '', options = {}) => {
        signal.throwIfAborted();
        const url = new URL(base);
        url.pathname = prefix + path;
        url.search = query;
        url.hash = '';
        const write = !['GET', 'HEAD'].includes(method) && options.readOnly !== true;
        if (write) submitted = true;
        const response = await fetchImpl(url, { method, headers: body === undefined ? headers : { ...headers, 'content-type': 'application/json' },
          ...(body === undefined ? {} : { body }), signal, redirect: 'error' });
        if (write && response.status >= 500) throw unknownOutcome(operationId);
        if (options.empty === true && response.status === 204) return { value: null, status: 204 };
        let value;
        const responseLimit = maxResponseBytes ?? (options.analysis && response.ok ? MAX_ANALYSIS_RESPONSE_BYTES : MAX_RESPONSE_BYTES);
        try { value = await readJson(response, options.privateResult ? '' : credential, responseLimit); }
        catch (error) { if (write && response.ok) throw unknownOutcome(operationId); throw error; }
        if (!response.ok) throw new BridgeFailure('tyloinfer/core-rejected', 'The core rejected this request', {
          http_status: response.status,
          ...(!options.privateResult && record(value?.error) ? { core_error: value.error } : {}),
          retryable: false,
        });
        return { value, status: response.status };
      };
      const health = (await request('/health')).value;
      if (health?.service !== 'tyloinfer-server' || health?.protocol_version !== '1.0') {
        throw new BridgeFailure('tyloinfer/core-incompatible', 'The endpoint is not a compatible TyloInfer core');
      }
      if (call.listenerChange) {
        const listener = (await request('/api/v1/runtime/listener')).value;
        if (listener?.configurable !== true) {
          throw new BridgeFailure('tyloinfer/operation-not-supported', 'The core listener is managed outside TyloInfer');
        }
        const { port, protocol } = call.listenerChange;
        if (protocol === 'openai' && port !== listener.port) {
          const currentPort = Number(base.port || (base.protocol === 'https:' ? 443 : 80));
          if (prefix || currentPort !== listener.port) {
            throw new BridgeFailure('tyloinfer/operation-not-supported', 'Changing the OpenAI port requires a direct connection to the core listener');
          }
          const next = new URL(base);
          next.port = String(port);
          if (!prepareProviderUpdate) throw new BridgeFailure('tyloinfer/operation-not-supported', 'The Host cannot update its core endpoint');
          providerUpdate = await prepareProviderUpdate(selected, { baseURL: next.toString() });
        }
      }
      if (call.authChange?.api_key) {
        if (!prepareProviderUpdate) throw new BridgeFailure('tyloinfer/operation-not-supported', 'The Host cannot save the core credential');
        providerUpdate = await prepareProviderUpdate(selected, { apiKey: call.authChange.api_key });
      } else if (call.authChange?.enabled && !credential) {
        throw new BridgeFailure('tyloinfer/bad-request', 'Enter an API key so Agent can reconnect');
      }
      if (call.kind === 'connection-info') {
        if (credential && base.toString().includes(credential)) {
          throw new BridgeFailure('tyloinfer/core-incompatible', 'Core address cannot expose credential material');
        }
        const listener = (await request('/api/v1/runtime/listener')).value;
        const port = listener?.anthropic_port ?? null;
        if (!record(listener) || port !== null && (!Number.isInteger(port) || port < 1 || port > 65535)) {
          throw new BridgeFailure('tyloinfer/core-invalid-response', 'The core returned invalid listener metadata');
        }
        const openai = new URL(base);
        openai.pathname = prefix + '/v1';
        const anthropic = port === null ? null : new URL(openai);
        if (anthropic) anthropic.port = String(port);
        return { ok: true, value: { base_url: base.toString(), openai_base_url: openai.toString(),
          anthropic_base_url: anthropic?.toString() ?? null, anthropic_port: port, protocol_version: health.protocol_version } };
      }
      if (Object.hasOwn(call, 'directoryId')) {
        const directory = (await request('/api/v1/models/directories', 'GET', undefined, new URLSearchParams({ directory_id: call.directoryId }).toString())).value;
        if (!record(directory) || directory.current_id !== call.directoryId || typeof directory.current_path !== 'string'
          || directory.current_path.length < 1 || directory.current_path.length > 4096 || /[\x00-\x1f\x7f]|^[A-Za-z]+:\/\//.test(directory.current_path)) {
          throw new BridgeFailure('tyloinfer/core-invalid-response', 'The core returned invalid directory metadata');
        }
        call.body = JSON.stringify({ prefix_directory: directory.current_path });
      }
      if (Object.hasOwn(call, 'companionIds')) {
        const paths = [];
        for (const id of call.companionIds) {
          let reference;
          try { reference = (await request('/api/v1/models/' + id + '/companion-reference')).value; }
          catch (error) {
            if (error instanceof BridgeFailure && error.code === 'tyloinfer/core-rejected' && error.details.http_status === 404 && !error.details.core_error) {
              throw new BridgeFailure('tyloinfer/operation-not-supported', 'This core cannot resolve registered GGUF companions; update the core before loading them', { retryable: false });
            }
            throw error;
          }
          if (!record(reference) || reference.model_id !== id || reference.format !== 'gguf' || typeof reference.path !== 'string'
            || reference.path.length < 1 || reference.path.length > 4096 || /[\x00-\x1f\x7f]|^[A-Za-z]+:\/\//.test(reference.path)) {
            throw new BridgeFailure('tyloinfer/core-invalid-response', 'The core returned an invalid registered GGUF reference');
          }
          paths.push(reference.path);
        }
        const body = JSON.parse(call.body);
        body.gguf_companions = paths;
        call.body = JSON.stringify(body);
        if (Buffer.byteLength(call.body) > MAX_REQUEST_BYTES) throw badRequest();
      }
      if (call.kind === 'official-local') {
        let response;
        try { response = await request(call.path); }
        catch (error) {
          if (error instanceof BridgeFailure && error.code === 'tyloinfer/core-rejected' && error.details.http_status === 404) {
            throw new BridgeFailure('tyloinfer/operation-not-supported', 'This core does not offer local official catalog snapshots', { retryable: false });
          }
          throw error;
        }
        if (response.status !== 200 || !call.validResult(response.value)) {
          throw new BridgeFailure('tyloinfer/core-invalid-response', 'The core returned invalid local catalog data');
        }
        return { ok: true, value: response.value };
      }
      if (call.kind === 'read') {
        const value = call.resource === 'health' ? health : (await request(call.path, 'GET', undefined, call.query, {
          analysis: call.resource === 'model_analysis',
        })).value;
        if (call.validResult && !call.validResult(value)) throw new BridgeFailure('tyloinfer/core-invalid-response', 'The core returned invalid maintenance data');
        return { ok: true, value };
      }
      if (call.kind === 'query') {
        if (Object.hasOwn(call, 'directoryGuard')) {
          const directory = (await request('/api/v1/hub/download-directory')).value;
          if (!record(directory) || typeof directory.actual_directory !== 'string') {
            throw new BridgeFailure('tyloinfer/core-invalid-response', 'The core returned invalid download directory state');
          }
          if (directory.actual_directory !== call.directoryGuard) throw badRequest();
        }
        const response = await request(call.path, call.method, call.body, '', { readOnly: true });
        if (response.status !== 200 || !call.validResult(response.value)) {
          throw new BridgeFailure('tyloinfer/core-invalid-response', 'The core returned invalid query data');
        }
        return { ok: true, value: response.value };
      }
      const requireJobHandler = async jobKinds => {
        const kinds = (await request('/api/v1/jobs/kinds')).value;
        if (!Array.isArray(kinds?.data)) throw new BridgeFailure('tyloinfer/core-invalid-response', 'The core returned an invalid operation catalog');
        if (!kinds.data.some(item => record(item) && jobKinds.has(item.kind))) {
          throw new BridgeFailure('tyloinfer/operation-not-supported', 'The core does not offer the selected operation handler');
        }
      };
      if (['download-retry', 'download-delete', 'job-delete'].includes(call.kind)) {
        const path = '/api/v1/jobs/' + call.id;
        const raw = (await request(path)).value;
        if (call.kind === 'job-delete' && (!record(raw) || typeof raw.kind !== 'string' || !/^[a-z][a-z0-9_.-]{0,127}$/.test(raw.kind))) {
          throw new BridgeFailure('tyloinfer/core-invalid-response', 'The core returned invalid job metadata');
        }
        const job = managedJob(raw, call.id, call.jobKinds ?? new Set([raw.kind]));
        const allowed = call.kind === 'download-retry' ? ['failed', 'cancelled', 'interrupted'] : ['succeeded', 'failed', 'cancelled', 'interrupted'];
        if (!allowed.includes(job.status)) {
          throw new BridgeFailure(call.kind === 'download-retry' ? 'tyloinfer/operation-not-retryable' : 'tyloinfer/operation-not-deletable',
            'This download state does not allow the selected action', { status: job.status });
        }
        if (call.kind !== 'download-retry') {
          const deleted = await request(path, 'DELETE', undefined, '', { empty: true });
          if (deleted.status !== 204) throw unknownOutcome(call.id);
          return { ok: true, value: null };
        }
        await requireJobHandler(new Set([job.kind]));
        const retried = await request(path + '/retry', 'POST');
        if (retried.status !== 202) throw unknownOutcome(call.id);
        try { managedJob(retried.value, null, new Set([job.kind])); }
        catch { throw unknownOutcome(call.id); }
        return { ok: true, value: retried.value };
      }
      if (call.kind === 'operation' || call.kind === 'cancel') {
        const path = '/api/v1/jobs/' + call.id;
        const job = managedJob((await request(path)).value, call.id, call.jobKinds);
        if (call.kind === 'operation') return { ok: true, value: job };
        if (job.cancel_requested || job.status === 'cancelling') return { ok: true, value: job };
        if (!CANCELLABLE.has(job.status)) throw new BridgeFailure('tyloinfer/operation-not-cancellable', 'This operation is already terminal', { status: job.status });
        let cancelled;
        try { cancelled = managedJob((await request(path + '/cancel', 'POST')).value, call.id, call.jobKinds); }
        catch (error) {
          if (submitted && error instanceof BridgeFailure && error.code === 'tyloinfer/operation-not-supported') throw unknownOutcome(operationId);
          throw error;
        }
        return { ok: true, value: cancelled };
      }
      if (call.format === 'job') await requireJobHandler(call.jobKinds);
      if (call.format === 'empty') {
        const response = await request(call.path, call.method, call.body, '', { empty: true });
        if (response.status !== 204) throw unknownOutcome(operationId);
        return { ok: true, value: null };
      }
      const accepted = await request(call.path, call.method, call.body, call.query, { privateResult: Boolean(call.authChange) });
      if (call.format === 'accepted') {
        if (accepted.status !== 202 || !record(accepted.value) || !uuid(accepted.value.operation_id) || accepted.value.status !== 'accepted') throw unknownOutcome(null);
      } else if (call.format === 'job') {
        if (accepted.status !== 202) throw unknownOutcome(null);
        try { managedJob(accepted.value, null, call.jobKinds); }
        catch { throw unknownOutcome(null); }
      } else if (accepted.status !== (call.status ?? 200) || !call.validResult(accepted.value)) throw unknownOutcome(null);
      if (providerUpdate) {
        try { await providerUpdate.commit(); }
        catch {
          throw new BridgeFailure('tyloinfer/provider-update-failed', 'The core accepted the change, but Agent could not save its connection settings',
            { ...(call.listenerChange ? { port: call.listenerChange.port } : {}), retryable: false });
        }
      }
      return { ok: true, value: call.authChange ? { enabled: accepted.value.enabled, has_key: accepted.value.has_key } : accepted.value };
    } catch (error) {
      if (providerUpdate && (!submitted || error instanceof BridgeFailure && error.code === 'tyloinfer/core-rejected')) {
        try { await providerUpdate.discard?.(); } catch { /* Leave the unused credential private if storage is unavailable. */ }
      }
      if (error instanceof BridgeFailure) return failure(error);
      if (submitted) return failure(unknownOutcome(operationId));
      return failure(new BridgeFailure(callerSignal?.aborted ? 'tyloinfer/cancelled' : 'tyloinfer/core-unavailable', 'Core request did not complete'));
    } finally {
      if (connectionChange) changingConnection = false;
    }
  };
}
