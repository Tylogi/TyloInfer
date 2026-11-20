const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const fields = (value, allowed, required = []) => object(value) && required.every(key => Object.hasOwn(value, key))
  && Object.keys(value).every(key => allowed.includes(key));
const text = (value, min = 1, max = 255) => typeof value === 'string' && value.length >= min && value.length <= max && !/[\x00-\x1f\x7f]/.test(value);
const integer = (value, min = 0, max = Number.MAX_SAFE_INTEGER) => Number.isSafeInteger(value) && value >= min && value <= max;
const uuid = value => typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value);
const hex = (value, size) => typeof value === 'string' && new RegExp('^[0-9a-f]{' + size + '}$').test(value);
const provider = value => ['huggingface', 'modelscope'].includes(value);
const repo = value => text(value, 3) && /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(value) && value.split('/').every(part => !['.', '..'].includes(part));
const optional = (value, key, valid) => !Object.hasOwn(value, key) || valid(value[key]);
const nullable = valid => value => value === null || valid(value);
const source = value => fields(value, ['provider', 'repo_id', 'revision'], ['provider', 'repo_id'])
  && provider(value.provider) && repo(value.repo_id) && optional(value, 'revision', nullable(value => text(value)));
const relative = value => text(value, 1, 1024) && !/^[\\/]|:/.test(value) && value.split(/[\\/]/).every(part => part !== '.' && part !== '..');
const downloadKinds = new Set(['download.huggingface', 'download.modelscope']);
const contextKinds = new Set(['runtime.context.configure']);
const budget = min => value => integer(value, min, 2 ** 50);
const qsa = value => fields(value, ['enabled', 'budget_bytes', 'ram_budget_bytes'])
  && optional(value, 'enabled', value => typeof value === 'boolean') && optional(value, 'budget_bytes', budget(1))
  && optional(value, 'ram_budget_bytes', nullable(budget(0)));
const quantization = value => fields(value, ['enabled', 'bits', 'algorithm'])
  && optional(value, 'enabled', value => typeof value === 'boolean') && optional(value, 'bits', value => [2, 2.5, 3, 3.5, 4, 6, 8].includes(value))
  && optional(value, 'algorithm', value => value === 'turboquant');
const memory = value => fields(value, ['qsa_mode', 'placement', 'allocation', 'gpu_budget_bytes', 'ram_preallocated', 'ram_budget_bytes'])
  && optional(value, 'qsa_mode', value => ['auto', 'fp16', 'affine8', 'tq4', 'legacy'].includes(value))
  && optional(value, 'placement', value => ['auto', 'auto_ssd', 'vram', 'ram'].includes(value))
  && optional(value, 'allocation', value => ['preallocated', 'dynamic'].includes(value))
  && optional(value, 'gpu_budget_bytes', nullable(budget(1))) && optional(value, 'ram_budget_bytes', nullable(budget(0)))
  && optional(value, 'ram_preallocated', value => typeof value === 'boolean')
  && (value.ram_preallocated !== false || value.placement === 'auto_ssd');
const serialize = value => {
  try { const result = JSON.stringify(value); return Buffer.byteLength(result) <= 65536 ? result : null; }
  catch { return null; }
};
const write = (path, request, validResult, method = 'POST') => {
  const body = serialize(request);
  return body === null ? null : { kind: 'write', path, body, method, format: 'value', validResult, maintenance: true };
};
const queryPost = (path, request, validResult, extra = {}) => {
  const body = serialize(request);
  return body === null ? null : { kind: 'query', path, body, method: 'POST', validResult, maintenance: true, ...extra };
};
const sameId = (value, id) => uuid(value) && value.toLowerCase() === id.toLowerCase();
const finite = value => typeof value === 'number' && Number.isFinite(value);
const yarn = value => object(value) && typeof value.supported === 'boolean' && typeof value.enabled === 'boolean'
  && nullable(value => integer(value, 1))(value.native_context) && nullable(value => integer(value, 1))(value.maximum_context)
  && nullable(value => finite(value) && value > 0)(value.maximum_factor) && finite(value.effective_factor) && value.effective_factor > 0;
const plan = (value, id) => object(value) && value.model_id === id && text(value.name) && hex(value.token, 64)
  && integer(value.total_bytes) && Array.isArray(value.files) && value.files.length > 0 && value.files.every(value => text(value, 1, 4096));

/** Resolve bounded model and download actions without accepting arbitrary methods or jobs. */
export function resolveModelAction(endpoint, payload) {
  if (endpoint === 'connection/info') {
    return fields(payload, []) ? { kind: 'connection-info' } : null;
  }
  if (endpoint === 'cache/configure-directory') {
    if (!fields(payload, ['directory_id', 'scope', 'confirmed'], ['directory_id', 'scope', 'confirmed']) || !hex(payload.directory_id, 32)
      || payload.scope !== 'all-loaded-models' || payload.confirmed !== true) return null;
    return { kind: 'write', path: '/api/v1/runtime/memory-policy', method: 'PUT', format: 'accepted', directoryId: payload.directory_id, maintenance: true };
  }
  if (endpoint === 'jobs/delete') {
    if (!fields(payload, ['job_id', 'confirmed'], ['job_id', 'confirmed']) || !uuid(payload.job_id) || payload.confirmed !== true) return null;
    return { kind: 'job-delete', id: payload.job_id, maintenance: true };
  }
  if (endpoint === 'jobs/clear-completed') {
    if (!fields(payload, ['scope', 'confirmed'], ['scope', 'confirmed']) || payload.scope !== 'all-completed' || payload.confirmed !== true) return null;
    return { kind: 'write', path: '/api/v1/jobs/completed', method: 'DELETE', format: 'empty', maintenance: true };
  }
  if (endpoint === 'hub/readme') {
    if (!fields(payload, ['provider', 'repo_id', 'revision'], ['provider', 'repo_id', 'revision'])
      || !provider(payload.provider) || !repo(payload.repo_id) || !text(payload.revision)) return null;
    return { kind: 'read', path: '/api/v1/hub/models/' + payload.provider + '/' + payload.repo_id.split('/').map(encodeURIComponent).join('/') + '/readme',
      query: new URLSearchParams({ revision: payload.revision }).toString(), maintenance: true,
      validResult: value => object(value) && (value.content === null || typeof value.content === 'string') };
  }
  if (endpoint === 'hub/probe-sources') {
    if (!fields(payload, ['sources'], ['sources']) || !Array.isArray(payload.sources) || payload.sources.length < 1 || payload.sources.length > 2
      || !Array.from(payload.sources).every(source)) return null;
    return queryPost('/api/v1/hub/download-sources/probe', payload,
      value => object(value) && Array.isArray(value.data) && value.data.length === payload.sources.length
        && value.data.every((row, index) => object(row) && row.provider === payload.sources[index].provider && row.repo_id === payload.sources[index].repo_id
          && finite(row.bytes_per_second) && row.bytes_per_second >= 0 && optional(row, 'route', value => ['direct', 'proxy'].includes(value))));
  }
  if (endpoint === 'hub/download-reuse') {
    if (!fields(payload, ['request'], ['request'])) return null;
    const request = payload.request;
    if (!fields(request, ['provider', 'repo_id', 'revision', 'destination', 'download_directory'], ['provider', 'repo_id', 'destination'])
      || !provider(request.provider) || !repo(request.repo_id) || !relative(request.destination)
      || !optional(request, 'revision', nullable(value => text(value)))
      || !optional(request, 'download_directory', nullable(value => text(value, 1, 4096)))) return null;
    return queryPost('/api/v1/hub/download-reuse', request,
      value => object(value) && object(value.reusable_files) && Object.values(value.reusable_files).every(value => integer(value)),
      { ...(typeof request.download_directory === 'string' ? { directoryGuard: request.download_directory } : {}) });
  }
  if (endpoint === 'instances/yarn') {
    if (!fields(payload, ['instance_id'], ['instance_id']) || !uuid(payload.instance_id)) return null;
    return { kind: 'read', path: '/api/v1/runtime/yarn/' + payload.instance_id, query: '', validResult: yarn };
  }
  if (endpoint === 'instances/configure') {
    if (!fields(payload, ['instance_id', 'request', 'confirmed'], ['instance_id', 'request', 'confirmed']) || !uuid(payload.instance_id) || payload.confirmed !== true
      || !fields(payload.request, ['pinned', 'idle_ttl_seconds']) || Object.keys(payload.request).length === 0
      || !optional(payload.request, 'pinned', value => typeof value === 'boolean')
      || !optional(payload.request, 'idle_ttl_seconds', nullable(value => integer(value)))) return null;
    return write('/api/v1/runtime/instances/' + payload.instance_id, payload.request,
      value => object(value) && sameId(value.id, payload.instance_id) && typeof value.pinned === 'boolean'
        && nullable(value => integer(value))(value.idle_ttl_seconds), 'PATCH');
  }
  if (endpoint === 'context/configure') {
    if (!fields(payload, ['request', 'confirmed'], ['request', 'confirmed']) || payload.confirmed !== true) return null;
    const request = payload.request;
    if (!fields(request, ['instance_id', 'context_size', 'yarn_enabled', 'qsa_kv_offload', 'kv_quantization', 'kv_memory'], ['instance_id', 'context_size'])
      || !uuid(request.instance_id) || !nullable(value => integer(value, 512, 2147483647))(request.context_size)
      || !optional(request, 'yarn_enabled', nullable(value => typeof value === 'boolean'))
      || !optional(request, 'qsa_kv_offload', nullable(qsa)) || !optional(request, 'kv_quantization', nullable(quantization))
      || !optional(request, 'kv_memory', nullable(memory))) return null;
    const body = serialize(request);
    return body === null ? null : { kind: 'write', path: '/api/v1/runtime/context', body, method: 'POST', format: 'job', jobKinds: contextKinds, maintenance: true };
  }
  if (endpoint === 'context/get') {
    if (!fields(payload, ['operation_id'], ['operation_id']) || !uuid(payload.operation_id)) return null;
    return { kind: 'operation', id: payload.operation_id, jobKinds: contextKinds };
  }
  if (endpoint === 'models/deletion-plan') {
    if (!fields(payload, ['model_id'], ['model_id']) || !hex(payload.model_id, 32)) return null;
    return { kind: 'read', path: '/api/v1/models/' + payload.model_id + '/deletion', query: '', maintenance: true,
      validResult: value => plan(value, payload.model_id) };
  }
  if (endpoint === 'models/delete') {
    if (!fields(payload, ['model_id', 'token', 'confirmed'], ['model_id', 'token', 'confirmed']) || !hex(payload.model_id, 32)
      || !hex(payload.token, 64) || payload.confirmed !== true) return null;
    return write('/api/v1/models/' + payload.model_id + '/delete', { token: payload.token },
      value => object(value) && integer(value.files) && integer(value.total_bytes));
  }
  if (endpoint === 'downloads/retry' || endpoint === 'downloads/delete') {
    if (!fields(payload, ['operation_id', 'confirmed'], ['operation_id', 'confirmed']) || !uuid(payload.operation_id) || payload.confirmed !== true) return null;
    return { kind: endpoint === 'downloads/retry' ? 'download-retry' : 'download-delete', id: payload.operation_id,
      jobKinds: downloadKinds, maintenance: true };
  }
  return null;
}
