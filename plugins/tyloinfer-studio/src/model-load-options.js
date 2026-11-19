export const defaultLoadOptions = () => ({ context: { context_size: null, automatic: true, jev_weights: 'stream', ssd_kv: false, ram_preallocated: true, vram_preallocated: true, kv_quantization: { enabled: false, bits: 4, algorithm: 'turboquant' } }, contextChanged: false, pin: false, ttl: null, devices: [], chunk: '' });
export function normalizeLoadOptions(value) {
  const result = defaultLoadOptions();
  if (!value || typeof value !== 'object') return result;
  if (typeof value.pin === 'boolean') result.pin = value.pin;
  if (Number.isSafeInteger(value.ttl) && value.ttl > 0) result.ttl = value.ttl;
  if (Array.isArray(value.devices) && value.devices.every(id => typeof id === 'string')) result.devices = value.devices;
  if (value.chunk !== '' && Number.isSafeInteger(Number(value.chunk)) && Number(value.chunk) >= 1 && Number(value.chunk) <= 8192) result.chunk = String(value.chunk);
  const context = value.context;
  if (!context || typeof context !== 'object') return result;
  if (Number.isSafeInteger(context.context_size) && context.context_size >= 512 && context.context_size <= 2147483647) result.context.context_size = context.context_size;
  for (const key of ['automatic', 'ssd_kv', 'ram_preallocated', 'vram_preallocated']) if (typeof context[key] === 'boolean') result.context[key] = context[key];
  if (['stream', 'vram'].includes(context.jev_weights)) result.context.jev_weights = context.jev_weights;
  if (typeof context.kv_quantization?.enabled === 'boolean' && [2, 2.5, 3, 3.5, 4, 6, 8].includes(context.kv_quantization.bits)) result.context.kv_quantization = { algorithm: 'turboquant', enabled: context.kv_quantization.enabled, bits: context.kv_quantization.bits };
  result.contextChanged = value.contextChanged === true;
  return result;
}
export function loadModelDrafts(storage) {
  try {
    const values = JSON.parse(storage().getItem('tyloinfer.model-load-options.v1') || '{}');
    return Object.fromEntries(Object.entries(values).filter(([id]) => /^[a-f0-9]{32}$/.test(id)).map(([id, value]) => [id, normalizeLoadOptions(value)]));
  } catch { return {}; }
}
export function saveModelDrafts(storage, values) {
  try { storage().setItem('tyloinfer.model-load-options.v1', JSON.stringify(values)); } catch { /* Current-page load options remain available without browser storage. */ }
}
export function nativeLoadRequest(model, options) {
  const request = { model: model.id, pin: options.pin };
  if (options.context.context_size != null) request.context_size = options.context.context_size;
  if (options.chunk !== '') request.prefill_chunk_size = Number(options.chunk);
  if (options.contextChanged) {
    const { automatic, ssd_kv, ram_preallocated, vram_preallocated, kv_quantization } = options.context;
    request.context_management = { automatic, ssd_kv, ram_preallocated, vram_preallocated, kv_quantization };
    if (model.structured_decisions) request.jev_weights = options.context.jev_weights;
  }
  if (options.devices.length) request.device_ids = options.devices;
  if (!options.pin && options.ttl != null) request.idle_ttl_seconds = options.ttl;
  return request;
}
