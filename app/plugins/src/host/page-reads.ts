import type { BridgeCall } from './types.js';
type Validator = (value: any) => boolean;
type FieldValidators = Record<string, Validator>;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const ARTIFACT_ID = /^[0-9a-f]{32}$/;
const NAMESPACE = /^[0-9a-f]{64}$/;
const object = (value: any): value is Record<string, any> => value !== null && typeof value === 'object' && !Array.isArray(value);
const text = (max: number): Validator => value => typeof value === 'string' && value.length <= max && !/[\x00-\x1f\x7f]/.test(value);
const integer = (min: number, max: number): Validator => value => Number.isSafeInteger(value) && value >= min && value <= max;
const oneOf = (values: readonly any[]): Validator => value => values.includes(value);
const uuid = (value: any) => typeof value === 'string' && UUID.test(value);
const identifier = (value: any) => typeof value === 'string' && ARTIFACT_ID.test(value);
const timestamp: Validator = value => typeof value === 'string' && value.length <= 40
  && /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)$/.test(value) && Number.isFinite(Date.parse(value));
const instance = { instance_id: uuid };
const states = ['queued', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled', 'interrupted'];
const reads: Record<string, readonly [string, FieldValidators]> = Object.freeze({
  health: ['/health', {}],
  status: ['/api/v1/runtime/status', instance],
  instances: ['/api/v1/runtime/instances', {}],
  models: ['/api/v1/models', { refresh: (value: any) => typeof value === 'boolean', include_components: (value: any) => typeof value === 'boolean' }],
  loaded_models: ['/api/v1/runtime/models', {}],
  resources: ['/api/v1/runtime/resources', {}],
  capabilities: ['/api/v1/runtime/capabilities', instance],
  listener: ['/api/v1/runtime/listener', {}],
  memory_policy: ['/api/v1/runtime/memory-policy', {}],
  inference_policy: ['/api/v1/runtime/inference-policy', {}],
  context_policy: ['/api/v1/runtime/context-policy', {}],
  model_aliases: ['/api/v1/runtime/model-aliases', {}],
  authentication_policy: ['/api/v1/auth/policy', {}],
  serving: ['/api/v1/runtime/serving', { scope: oneOf(['session', 'alltime']), model: text(255) }],
  metrics: ['/api/v1/runtime/metrics', { ...instance, since: timestamp, limit: integer(1, 2000) }],
  logs: ['/api/v1/runtime/logs', { ...instance, level: oneOf(['debug', 'info', 'warning', 'error']), after: integer(0, Number.MAX_SAFE_INTEGER), limit: integer(1, 2000) }],
  jobs: ['/api/v1/jobs', { status: oneOf(states), kind: (value: any) => typeof value === 'string' && /^[a-z][a-z0-9_.-]{0,63}$/.test(value), limit: integer(1, 200), offset: integer(0, Number.MAX_SAFE_INTEGER) }],
  job_kinds: ['/api/v1/jobs/kinds', {}],
  download_directory: ['/api/v1/hub/download-directory', {}],
  model_directories: ['/api/v1/models/directories', { directory_id: identifier }],
  prefix_cache: ['/api/v1/runtime/cache/entries', {
    namespace: (value: any) => typeof value === 'string' && NAMESPACE.test(value), offset: integer(0, Number.MAX_SAFE_INTEGER), limit: integer(1, 200),
    query: text(200), search_in: oneOf(['all', 'text', 'id']), text_filter: oneOf(['all', 'saved', 'missing']),
    chain_filter: oneOf(['all', 'complete', 'incomplete']), sort: oneOf(['recent', 'oldest', 'length', 'size']),
  }],
});

/** Resolve a fixed core GET with resource-specific named filters, never a client URL. */
export function resolveCoreRead(payload: any): BridgeCall | null {
  if (!object(payload) || !Object.hasOwn(payload, 'resource') || typeof payload.resource !== 'string'
    || !Object.keys(payload).every(key => ['resource', 'params'].includes(key)) || !Object.hasOwn(reads, payload.resource)) return null;
  const [path, fields] = reads[payload.resource]!;
  const params = Object.hasOwn(payload, 'params') ? payload.params : {};
  if (!object(params)) return null;
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (!Object.hasOwn(fields, key) || !fields[key]!(value)) return null;
    query.set(key, String(value));
  }
  return { kind: 'read', resource: payload.resource, path, query: query.toString() };
}

/** Resolve analysis by the core's catalog artifact id, not a path or model name. */
export function resolveModelAnalysis(payload: any): BridgeCall | null {
  if (!object(payload) || Object.keys(payload).length !== 1 || !identifier(payload.model_id)) return null;
  return { kind: 'read', resource: 'model_analysis', path: '/api/v1/models/' + payload.model_id + '/analysis', query: '' };
}

/** Resolve local model metadata by catalog identity without accepting filesystem inputs. */
export function resolveModelDetails(endpoint: string, payload: any): BridgeCall | null {
  if (endpoint === 'models/load-plan') {
    if (!object(payload) || !identifier(payload.model_id)) return null;
    const validators: FieldValidators = {
      reserve: (value: any) => [5, 10, 20].includes(value),
      context_size: (value: any) => Number.isSafeInteger(value) && value >= 512 && value <= 2147483647,
      precision: (value: any) => [2, 2.5, 3, 3.5, 4, 6, 8, 16].includes(value),
      preallocate: (value: any) => typeof value === 'boolean',
      streamed_jev: (value: any) => typeof value === 'boolean',
      device: (value: any) => typeof value === 'string' && /^[a-z0-9_-]{1,64}$/i.test(value),
    };
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(payload)) {
      if (key === 'model_id') continue;
      if (!Object.hasOwn(validators, key) || !validators[key]!(value)) return null;
      query.set(key, String(value));
    }
    return { kind: 'read', resource: 'model_load_plan', path: '/api/v1/models/' + payload.model_id + '/load-plan', query: query.toString() };
  }
  const profile = endpoint === 'models/cache-profile';
  if (!profile && endpoint !== 'models/directory') return null;
  const allowed = profile ? ['model_id', 'include_recurrent'] : ['model_id'];
  if (!object(payload) || !Object.hasOwn(payload, 'model_id') || !identifier(payload.model_id)
    || !Object.keys(payload).every(key => allowed.includes(key))
    || (Object.hasOwn(payload, 'include_recurrent') && typeof payload.include_recurrent !== 'boolean')) return null;
  const query = new URLSearchParams();
  if (Object.hasOwn(payload, 'include_recurrent')) query.set('include_recurrent', String(payload.include_recurrent));
  return { kind: 'read', resource: profile ? 'model_cache_profile' : 'model_directory',
    path: '/api/v1/models/' + payload.model_id + (profile ? '/cache-profile' : '/directory'), query: query.toString() };
}
