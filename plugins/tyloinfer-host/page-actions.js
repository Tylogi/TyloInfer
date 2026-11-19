const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const fields = (value, allowed, required = []) => object(value) && required.every(key => Object.hasOwn(value, key))
  && Object.keys(value).every(key => allowed.includes(key));
const text = (value, min, max) => typeof value === 'string' && value.length >= min && value.length <= max && !/[\x00-\x1f\x7f]/.test(value);
const integer = (value, min, max = Number.MAX_SAFE_INTEGER) => Number.isSafeInteger(value) && value >= min && value <= max;
const provider = value => ['huggingface', 'modelscope'].includes(value);
const repo = value => text(value, 3, 255) && /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(value)
  && value.split('/').every(part => !['.', '..'].includes(part));
const uuid = value => typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value);
const directoryId = value => typeof value === 'string' && /^[0-9a-f]{32}$/.test(value);
const downloadKinds = new Set(['download.huggingface', 'download.modelscope']);
const serviceKinds = new Set(['runtime.memory.configure']);
const body = value => {
  try {
    const serialized = JSON.stringify(value);
    return Buffer.byteLength(serialized) <= 65536 ? serialized : null;
  } catch { return null; }
};
const query = params => new URLSearchParams(Object.entries(params).filter(([, value]) => value !== undefined).map(([key, value]) => [key, String(value)])).toString();
const aliases = value => object(value) && Object.keys(value).length <= 128
  && Object.entries(value).every(([name, alias]) => text(name, 1, 255) && text(alias, 0, 255));
const service = Object.freeze({
  inference_policy: { path: 'inference-policy', valid: request => fields(request, ['mtp_enabled'], ['mtp_enabled']) && typeof request.mtp_enabled === 'boolean',
    result: value => object(value) && typeof value.mtp_enabled === 'boolean' },
  model_aliases: { path: 'model-aliases', valid: request => fields(request, ['aliases'], ['aliases']) && aliases(request.aliases),
    result: value => object(value) && aliases(value.aliases) },
  context_policy: { path: 'context-policy', valid: request => fields(request, ['max_context_size'], ['max_context_size'])
      && (request.max_context_size === null || integer(request.max_context_size, 512, 2147483647)),
    result: value => object(value) && (value.max_context_size === null || integer(value.max_context_size, 512, 2147483647)) },
  memory_policy: { path: 'memory-policy', valid: request => fields(request,
    ['total_limit_bytes', 'model_limit_bytes', 'prefix_limit_bytes', 'prefix_disk_limit_bytes', 'prefix_ram_limit_bytes'])
    && Object.keys(request).length > 0 && Object.entries(request).every(([key, value]) => value === null || integer(value, key.startsWith('prefix_') ? 0 : 1)) },
});

/** Resolve explicit page actions into fixed routes without exposing arbitrary jobs or methods. */
export function resolvePageAction(endpoint, payload) {
  if (endpoint === 'hub/search') {
    if (!fields(payload, ['provider', 'query', 'formats', 'sort', 'direction', 'cursor', 'limit'], ['provider', 'query'])
      || !provider(payload.provider) || !text(payload.query, 0, 255)
      || (Object.hasOwn(payload, 'formats') && (!Array.isArray(payload.formats) || payload.formats.length > 3 || !payload.formats.every(format => ['tq', 'gguf', 'mlx'].includes(format))))
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
  if (endpoint === 'models/register') {
    if (!fields(payload, ['directory_id'], ['directory_id']) || !directoryId(payload.directory_id)) return null;
    return { kind: 'write', path: '/api/v1/models/directories/register', body: body(payload), method: 'POST', format: 'value',
      validResult: value => object(value) && Array.isArray(value.data) };
  }
  if (endpoint === 'downloads/create') {
    if (!fields(payload, ['provider', 'request'], ['provider', 'request']) || !provider(payload.provider)) return null;
    const request = payload.request;
    if (!fields(request, ['repo_id', 'destination', 'revision', 'include', 'expected_bytes', 'max_workers', 'download_directory'], ['repo_id', 'destination'])
      || !repo(request.repo_id) || !text(request.destination, 1, 1024) || /^[\\/]|^[A-Za-z]:/.test(request.destination)
      || request.destination.split(/[\\/]/).some(part => part === '..')
      || (Object.hasOwn(request, 'revision') && !text(request.revision, 1, 255))
      || (Object.hasOwn(request, 'download_directory') && !text(request.download_directory, 1, 4096))
      || (Object.hasOwn(request, 'include') && (!Array.isArray(request.include) || request.include.length > 4096 || !request.include.every(pattern => text(pattern, 1, 1024))))
      || (Object.hasOwn(request, 'expected_bytes') && request.expected_bytes !== null && !integer(request.expected_bytes, 0))
      || (Object.hasOwn(request, 'max_workers') && !integer(request.max_workers, 1, 16))) return null;
    const kind = 'download.' + payload.provider;
    const serialized = body({ kind, payload: request });
    if (!serialized) return null;
    return { kind: 'write', path: '/api/v1/jobs', body: serialized, method: 'POST', format: 'job', jobKinds: new Set([kind]) };
  }
  for (const [prefix, jobKinds] of [['downloads', downloadKinds], ['service', serviceKinds]]) {
    if ([prefix + '/get', prefix + '/cancel'].includes(endpoint)) {
      if (!fields(payload, ['operation_id'], ['operation_id']) || !uuid(payload.operation_id)) return null;
      return { kind: endpoint.endsWith('/get') ? 'operation' : 'cancel', id: payload.operation_id, jobKinds };
    }
  }
  if (endpoint === 'service/configure') {
    if (!fields(payload, ['resource', 'request'], ['resource', 'request']) || typeof payload.resource !== 'string'
      || !Object.hasOwn(service, payload.resource)) return null;
    const route = service[payload.resource];
    if (!route.valid(payload.request)) return null;
    const serialized = body(payload.request);
    if (!serialized) return null;
    return { kind: 'write', path: '/api/v1/runtime/' + route.path, body: serialized, method: 'PUT',
      format: payload.resource === 'memory_policy' ? 'accepted' : 'value', validResult: route.result };
  }
  return null;
}
