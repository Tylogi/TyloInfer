import type { BridgeCall } from './types.js';
const object = (value: any): value is Record<string, any> => value !== null && typeof value === 'object' && !Array.isArray(value);
const fields = (value: any, allowed: readonly string[], required: readonly string[] = []) => object(value) && required.every(key => Object.hasOwn(value, key))
  && Object.keys(value).every(key => allowed.includes(key));
const text = (value: any, min: number, max: number) => typeof value === 'string' && value.length >= min && value.length <= max && !/[\x00-\x1f\x7f]/.test(value);
const integer = (value: any, min: number, max = Number.MAX_SAFE_INTEGER) => Number.isSafeInteger(value) && value >= min && value <= max;
const provider = (value: any) => ['huggingface', 'modelscope'].includes(value);
const repo = (value: any) => text(value, 3, 255) && /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(value)
  && (value as string).split('/').every((part: string) => !['.', '..'].includes(part));
const uuid = (value: any) => typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value);
const directoryId = (value: any) => typeof value === 'string' && /^[0-9a-f]{32}$/.test(value);
const nodeUrl = (value: any) => {
  if (!text(value, 1, 2048) || !/^https?:\/\/[^/\s]+/.test(value) || /\s/.test(value)) return false;
  try {
    const url = new URL(value);
    return !url.username && !url.password && !url.search && !url.hash;
  } catch { return false; }
};
const remoteNode = (value: any) => object(value) && uuid(value.id) && text(value.name, 1, 64) && nodeUrl(value.url)
  && typeof value.healthy === 'boolean' && Array.isArray(value.models) && integer(value.active_requests, 0) && object(value.metrics);
const downloadKinds = new Set(['download.huggingface', 'download.modelscope']);
const serviceKinds = new Set(['runtime.memory.configure']);
const body = (value: any) => {
  try {
    const serialized = JSON.stringify(value);
    return Buffer.byteLength(serialized) <= 65536 ? serialized : null;
  } catch { return null; }
};
const query = (params: Record<string, any>) => new URLSearchParams(Object.entries(params).filter(([, value]) => value !== undefined).map(([key, value]) => [key, String(value)] as [string, string])).toString();
const aliases = (value: any) => object(value) && Object.keys(value).length <= 128
  && Object.entries(value).every(([name, alias]) => text(name, 1, 255) && text(alias, 0, 255));
const service: Record<string, { path: string; valid: (request: any) => boolean; result?: (value: any) => boolean }> = Object.freeze({
  inference_policy: { path: 'inference-policy', valid: (request: any) => fields(request, ['mtp_enabled'], ['mtp_enabled']) && typeof request.mtp_enabled === 'boolean',
    result: (value: any) => object(value) && typeof value.mtp_enabled === 'boolean' },
  model_aliases: { path: 'model-aliases', valid: (request: any) => fields(request, ['aliases'], ['aliases']) && aliases(request.aliases),
    result: (value: any) => object(value) && aliases(value.aliases) },
  context_policy: { path: 'context-policy', valid: (request: any) => fields(request, ['max_context_size'], ['max_context_size'])
      && (request.max_context_size === null || integer(request.max_context_size, 512, 2147483647)),
    result: (value: any) => object(value) && (value.max_context_size === null || integer(value.max_context_size, 512, 2147483647)) },
  memory_policy: { path: 'memory-policy', valid: (request: any) => fields(request,
    ['total_limit_bytes', 'model_limit_bytes', 'prefix_limit_bytes', 'prefix_disk_limit_bytes', 'prefix_ram_limit_bytes'])
    && Object.keys(request).length > 0 && Object.entries(request).every(([key, value]) => value === null || integer(value, key.startsWith('prefix_') ? 0 : 1)) },
});

/** Resolve explicit page actions into fixed routes without exposing arbitrary jobs or methods. */
export function resolvePageAction(endpoint: string, payload: any): BridgeCall | null {
  if (endpoint === 'service/configure-listener') {
    if (!fields(payload, ['port', 'protocol'], ['port', 'protocol']) || !integer(payload.port, 1, 65535)
      || !['openai', 'anthropic'].includes(payload.protocol)) return null;
    return { kind: 'write', path: '/api/v1/runtime/listener', method: 'PUT', body: body(payload), format: 'value', maintenance: true,
      listenerChange: payload, validResult: value => object(value) && value[payload.protocol === 'openai' ? 'port' : 'anthropic_port'] === payload.port };
  }
  if (endpoint === 'service/configure-authentication') {
    if (!fields(payload, ['enabled', 'api_key'], ['enabled']) || typeof payload.enabled !== 'boolean'
      || (Object.hasOwn(payload, 'api_key') && (!payload.enabled || typeof payload.api_key !== 'string' || !/^[\x21-\x7e]{1,512}$/.test(payload.api_key)))) return null;
    return { kind: 'write', path: '/api/v1/auth/policy', method: 'PUT', body: body(payload), format: 'value', maintenance: true,
      authChange: payload, validResult: value => object(value) && value.enabled === payload.enabled && typeof value.has_key === 'boolean' };
  }
  if (endpoint === 'service/remote-nodes') {
    if (!fields(payload, ['refresh'], ['refresh']) || typeof payload.refresh !== 'boolean') return null;
    return { kind: 'read', path: '/api/v1/cluster/nodes', query: query(payload), maintenance: payload.refresh,
      validResult: value => object(value) && Array.isArray(value.data) && value.data.every(remoteNode) };
  }
  if (endpoint === 'service/add-remote-node') {
    if (!fields(payload, ['name', 'url', 'api_key_env', 'enabled'], ['name', 'url', 'enabled'])
      || !text(payload.name, 1, 64) || !/^[A-Za-z_][\w.-]*$/.test(payload.name) || !nodeUrl(payload.url) || typeof payload.enabled !== 'boolean'
      || (payload.api_key_env != null && (!text(payload.api_key_env, 1, 255) || !/^[A-Za-z_]\w*$/.test(payload.api_key_env)))) return null;
    return { kind: 'write', path: '/api/v1/cluster/nodes', method: 'POST', body: body(payload),
      format: 'value', status: 201, validResult: remoteNode, maintenance: true };
  }
  if (endpoint === 'service/delete-remote-node') {
    if (!fields(payload, ['id'], ['id']) || !uuid(payload.id)) return null;
    return { kind: 'write', path: '/api/v1/cluster/nodes/' + payload.id, method: 'DELETE', format: 'empty' };
  }
  if (endpoint === 'hub/search') {
    if (!fields(payload, ['provider', 'query', 'formats', 'sort', 'direction', 'cursor', 'limit'], ['provider', 'query'])
      || !provider(payload.provider) || !text(payload.query, 0, 255)
      || (Object.hasOwn(payload, 'formats') && (!Array.isArray(payload.formats) || payload.formats.length > 3 || !payload.formats.every((format: string) => ['tq', 'gguf', 'mlx'].includes(format))))
      || (Object.hasOwn(payload, 'sort') && !['relevance', 'downloads', 'likes', 'updated'].includes(payload.sort))
      || (Object.hasOwn(payload, 'direction') && !['asc', 'desc'].includes(payload.direction))
      || (Object.hasOwn(payload, 'cursor') && (typeof payload.cursor !== 'string' || !/^[A-Za-z0-9_-]{16,64}$/.test(payload.cursor)))
      || (Object.hasOwn(payload, 'limit') && !integer(payload.limit, 1, 100))) return null;
    return { kind: 'read', path: '/api/v1/hub/models', query: query({ provider: payload.provider, query: payload.query,
      ...(payload.formats?.length ? { formats: payload.formats.join(',') } : {}), sort: payload.sort, direction: payload.direction,
      cursor: payload.cursor, limit: payload.limit, browse: true }) };
  }
  if (endpoint === 'hub/info') {
    if (!fields(payload, ['provider', 'repo_id', 'revision'], ['provider', 'repo_id']) || !provider(payload.provider) || !repo(payload.repo_id)
      || (Object.hasOwn(payload, 'revision') && !text(payload.revision, 1, 255))) return null;
    return { kind: 'read', path: '/api/v1/hub/models/' + payload.provider + '/' + payload.repo_id.split('/').map(encodeURIComponent).join('/'), query: query({ revision: payload.revision }) };
  }
  if (endpoint === 'hub/official-local') {
    if (!fields(payload, [])) return null;
    return { kind: 'official-local', path: '/api/v1/hub/official/local',
      validResult: value => object(value) && object(value.system) && text(value.system.platform, 1, 64)
        && text(value.system.machine, 1, 64) && ['metal', 'cuda', 'rocm', 'cpu', 'unknown'].includes(value.system.backend)
        && Array.isArray(value.data) && value.data.every(model => object(model) && text(model.id, 1, 128)
          && text(model.name, 1, 255) && Array.isArray(model.sources) && Array.isArray(model.variants))
        && typeof value.refreshing === 'boolean' };
  }
  if (endpoint === 'hub/official') {
    if (!fields(payload, ['refresh']) || (Object.hasOwn(payload, 'refresh') && typeof payload.refresh !== 'boolean')) return null;
    return { kind: 'read', path: '/api/v1/hub/official', query: query(payload) };
  }
  if (endpoint === 'models/browse-directory') {
    if (!fields(payload, ['path'], ['path']) || !text(payload.path, 1, 4096)
      || !/^(?:\/(?!\/)|[A-Za-z]:[\\/])/.test(payload.path)) return null;
    return { kind: 'read', path: '/api/v1/models/directories', query: query(payload),
      validResult: value => object(value) && directoryId(value.current_id) && Array.isArray(value.data) };
  }
  if (endpoint === 'models/open-directory') {
    if (!fields(payload, ['directory_id'], ['directory_id']) || !directoryId(payload.directory_id)) return null;
    return { kind: 'write', path: '/api/v1/models/directories/open', method: 'POST', body: body(payload),
      format: 'value', validResult: value => object(value) && value.opened === true };
  }
  if (endpoint === 'models/register') {
    if (!fields(payload, ['directory_id', 'file_name'], ['directory_id']) || !directoryId(payload.directory_id)) return null;
    if (Object.hasOwn(payload, 'file_name') && (!text(payload.file_name, 1, 512)
      || /[\\/:]/.test(payload.file_name) || ['.', '..'].includes(payload.file_name))) return null;
    return { kind: 'write', path: '/api/v1/models/directories/register', query: 'allow_empty=true', body: body(payload), method: 'POST', format: 'value',
      maintenance: true, validResult: value => object(value) && Array.isArray(value.data) };
  }
  if (endpoint === 'downloads/create') {
    if (!fields(payload, ['provider', 'request'], ['provider', 'request']) || !provider(payload.provider)) return null;
    const request = payload.request;
    if (!fields(request, ['repo_id', 'destination', 'revision', 'include', 'expected_bytes', 'max_workers', 'download_directory'], ['repo_id', 'destination'])
      || !repo(request.repo_id) || !text(request.destination, 1, 1024) || /^[\\/]|^[A-Za-z]:/.test(request.destination)
      || request.destination.split(/[\\/]/).some((part: string) => part === '..')
      || (Object.hasOwn(request, 'revision') && !text(request.revision, 1, 255))
      || (Object.hasOwn(request, 'download_directory') && !text(request.download_directory, 1, 4096))
      || (Object.hasOwn(request, 'include') && (!Array.isArray(request.include) || request.include.length > 4096 || !request.include.every((pattern: any) => text(pattern, 1, 1024))))
      || (Object.hasOwn(request, 'expected_bytes') && request.expected_bytes !== null && !integer(request.expected_bytes, 0))
      || (Object.hasOwn(request, 'max_workers') && !integer(request.max_workers, 1, 16))) return null;
    const kind = 'download.' + payload.provider;
    const serialized = body({ kind, payload: request });
    if (!serialized) return null;
    return { kind: 'write', path: '/api/v1/jobs', body: serialized, method: 'POST', format: 'job', jobKinds: new Set([kind]) };
  }
  for (const [prefix, jobKinds] of [['downloads', downloadKinds], ['service', serviceKinds]] as [string, Set<string>][]) {
    if ([prefix + '/get', prefix + '/cancel'].includes(endpoint)) {
      if (!fields(payload, ['operation_id'], ['operation_id']) || !uuid(payload.operation_id)) return null;
      return { kind: endpoint.endsWith('/get') ? 'operation' : 'cancel', id: payload.operation_id, jobKinds };
    }
  }
  if (endpoint === 'service/configure') {
    if (!fields(payload, ['resource', 'request'], ['resource', 'request']) || typeof payload.resource !== 'string'
      || !Object.hasOwn(service, payload.resource)) return null;
    const route = service[payload.resource]!;
    if (!route.valid(payload.request)) return null;
    const serialized = body(payload.request);
    if (!serialized) return null;
    return { kind: 'write', path: '/api/v1/runtime/' + route.path, body: serialized, method: 'PUT',
      format: payload.resource === 'memory_policy' ? 'accepted' : 'value', validResult: route.result };
  }
  return null;
}
