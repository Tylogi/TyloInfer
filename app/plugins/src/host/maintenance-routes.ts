import type { BridgeCall } from './types.js';
const object = (value: any): value is Record<string, any> => value !== null && typeof value === 'object' && !Array.isArray(value);
const fields = (value: any, allowed: readonly string[], required: readonly string[] = []) => object(value) && required.every(key => Object.hasOwn(value, key))
  && Object.keys(value).every(key => allowed.includes(key));
const integer = (value: any, max = Number.MAX_SAFE_INTEGER) => Number.isSafeInteger(value) && value >= 0 && value <= max;
const uuid = (value: any) => typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value);
const hash = (value: any, size: number) => typeof value === 'string' && new RegExp('^[0-9a-f]{' + size + '}$').test(value);
const text = (value: any) => typeof value === 'string' && value.length > 0;
const states = ['queued', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled', 'interrupted'];
const query = (value: Record<string, any>) => new URLSearchParams(Object.entries(value).map(([key, item]) => [key, String(item)] as [string, string])).toString();
const counters = (value: any, keys: readonly string[]) => object(value) && value.status === 'ok' && keys.every(key => integer(value[key]));
const write = (path: string, body: any, validResult: (value: any) => boolean, method = 'POST'): BridgeCall => ({ kind: 'write', path, body: JSON.stringify(body), method, format: 'value', validResult, maintenance: true });
const date = (value: any) => typeof value === 'string' && Number.isFinite(Date.parse(value));
const job = (value: any, id: string) => object(value) && uuid(value.id) && value.id.toLowerCase() === id.toLowerCase()
  && text(value.kind) && states.includes(value.status) && object(value.payload) && object(value.progress_data)
  && typeof value.progress === 'number' && Number.isFinite(value.progress) && value.progress >= 0 && value.progress <= 1
  && typeof value.cancel_requested === 'boolean' && (value.result === null || object(value.result))
  && (value.error === null || object(value.error)) && date(value.created_at) && date(value.updated_at)
  && (value.started_at === null || date(value.started_at)) && (value.completed_at === null || date(value.completed_at));
const events = (value: any, id: string, after: number) => {
  if (!object(value) || !Array.isArray(value.data)) return false;
  let previous = after;
  for (const item of value.data) {
    if (!object(item) || !uuid(item.job_id) || item.job_id.toLowerCase() !== id.toLowerCase() || !integer(item.sequence)
      || item.sequence <= previous || !text(item.type) || !text(item.level) || (item.message !== null && typeof item.message !== 'string')
      || !object(item.data) || !date(item.created_at)) return false;
    previous = item.sequence;
  }
  return true;
};
const cacheText = (value: any) => object(value) && (value.available === false ? text(value.reason)
  : value.available === true && typeof value.text === 'string' && integer(value.total_tokens) && integer(value.offset)
    && (value.next_offset === null || integer(value.next_offset) && value.next_offset > value.offset && value.next_offset < value.total_tokens));
const directoryStatus = (value: any) => object(value) && (value.directory === null || text(value.directory)) && text(value.actual_directory)
  && text(value.default_directory) && typeof value.evict_prefix_cache === 'boolean'
  && ['disk_total_bytes', 'disk_free_bytes'].every(key => value[key] === null || integer(value[key]))
  && ['disk_available_bytes', 'download_reserved_bytes', 'reclaimable_prefix_bytes'].every(key => !Object.hasOwn(value, key) || value[key] === null || integer(value[key]));

/** Resolve confirmed maintenance operations and owner-only details into fixed core routes. */
export function resolveMaintenanceCall(endpoint: string, payload: any): BridgeCall | null {
  if (endpoint === 'cache/clear') {
    if (!fields(payload, ['instance_id', 'confirmed'], ['instance_id', 'confirmed']) || !uuid(payload.instance_id) || payload.confirmed !== true) return null;
    return write('/api/v1/runtime/cache/clear', { instance_id: payload.instance_id },
      value => counters(value, ['released_snapshots', 'image_decode_cache_released_bytes']));
  }
  if (endpoint === 'cache/trim') {
    if (!fields(payload, ['instance_id', 'target_bytes', 'confirmed'], ['instance_id', 'target_bytes', 'confirmed'])
      || !uuid(payload.instance_id) || !integer(payload.target_bytes) || payload.confirmed !== true) return null;
    return write('/api/v1/runtime/cache/trim', { instance_id: payload.instance_id, target_bytes: payload.target_bytes },
      value => counters(value, ['released_bytes', 'target_bytes']) && value.target_bytes === payload.target_bytes);
  }
  if (endpoint === 'cache/purge') {
    if (!fields(payload, ['scope', 'namespace', 'confirmed'], ['scope', 'confirmed']) || payload.confirmed !== true) return null;
    if (payload.scope === 'namespace' ? !hash(payload.namespace, 64) : payload.scope !== 'all' || Object.hasOwn(payload, 'namespace')) return null;
    return write('/api/v1/runtime/cache/purge', { namespace: payload.scope === 'all' ? null : payload.namespace },
      value => counters(value, ['released_bytes', 'removed_blocks', 'failed_blocks']));
  }
  if (endpoint === 'cache/text') {
    if (!fields(payload, ['namespace', 'block', 'offset', 'limit'], ['namespace', 'block']) || !hash(payload.namespace, 64) || !hash(payload.block, 64)
      || (Object.hasOwn(payload, 'offset') && !integer(payload.offset))
      || (Object.hasOwn(payload, 'limit') && (!integer(payload.limit, 8192) || payload.limit < 1))) return null;
    const params = Object.fromEntries(['offset', 'limit'].filter(key => Object.hasOwn(payload, key)).map(key => [key, payload[key]]));
    return { kind: 'read', path: '/api/v1/runtime/cache/entries/' + payload.namespace + '/' + payload.block + '/text', query: query(params), validResult: cacheText, maintenance: true };
  }
  if (endpoint === 'downloads/configure-directory') {
    if (!fields(payload, ['request', 'confirmed'], ['request', 'confirmed']) || payload.confirmed !== true
      || !fields(payload.request, ['directory_id', 'evict_prefix_cache']) || Object.keys(payload.request).length === 0
      || (Object.hasOwn(payload.request, 'directory_id') && payload.request.directory_id !== null && !hash(payload.request.directory_id, 32))
      || (Object.hasOwn(payload.request, 'evict_prefix_cache') && typeof payload.request.evict_prefix_cache !== 'boolean')) return null;
    return write('/api/v1/hub/download-directory', payload.request, directoryStatus, 'PUT');
  }
  if (endpoint === 'jobs/get') {
    if (!fields(payload, ['job_id'], ['job_id']) || !uuid(payload.job_id)) return null;
    return { kind: 'read', path: '/api/v1/jobs/' + payload.job_id, query: '', validResult: value => job(value, payload.job_id) };
  }
  if (endpoint === 'jobs/events') {
    if (!fields(payload, ['job_id', 'after', 'limit', 'tail'], ['job_id']) || !uuid(payload.job_id)
      || (Object.hasOwn(payload, 'after') && !integer(payload.after))
      || (Object.hasOwn(payload, 'limit') && (!integer(payload.limit, 1000) || payload.limit < 1))
      || (Object.hasOwn(payload, 'tail') && typeof payload.tail !== 'boolean')) return null;
    const params = Object.fromEntries(['after', 'limit', 'tail'].filter(key => Object.hasOwn(payload, key)).map(key => [key, payload[key]]));
    return { kind: 'read', path: '/api/v1/jobs/' + payload.job_id + '/events', query: query(params), validResult: value => events(value, payload.job_id, payload.after ?? 0) };
  }
  return null;
}
