import type { StorageLike } from './types.ts';

export interface LoadOptions {
  context: { context_size: number | null; reserve_percent: number; precision: number | null; jev_weights: string; vram_preallocated: boolean };
  pin: boolean;
  ttl: number | null;
  devices: string[];
}

export const defaultLoadOptions = (): LoadOptions => ({ context: { context_size: null, reserve_percent: 10, precision: null, jev_weights: 'stream', vram_preallocated: true }, pin: false, ttl: null, devices: [] });
export function normalizeLoadOptions(value: any): LoadOptions {
  const result = defaultLoadOptions();
  if (!value || typeof value !== 'object') return result;
  if (typeof value.pin === 'boolean') result.pin = value.pin;
  if (Number.isSafeInteger(value.ttl) && value.ttl > 0) result.ttl = value.ttl;
  if (Array.isArray(value.devices) && value.devices.every((id: unknown) => typeof id === 'string')) result.devices = value.devices;
  const context = value.context;
  if (!context || typeof context !== 'object') return result;
  if (Number.isSafeInteger(context.context_size) && context.context_size >= 512 && context.context_size <= 2147483647) result.context.context_size = context.context_size;
  if ([5, 10, 20].includes(context.reserve_percent)) result.context.reserve_percent = context.reserve_percent;
  if ([2, 2.5, 3, 3.5, 4, 6, 8].includes(context.precision)) result.context.precision = context.precision;
  if (typeof context.vram_preallocated === 'boolean' && (Object.hasOwn(context, 'reserve_percent') || context.automatic === false))
    result.context.vram_preallocated = context.vram_preallocated;
  if (['stream', 'vram'].includes(context.jev_weights)) result.context.jev_weights = context.jev_weights;
  if (!Object.hasOwn(context, 'precision') && context.automatic === false && context.kv_quantization?.enabled === true
      && [2, 2.5, 3, 3.5, 4, 6, 8].includes(context.kv_quantization.bits))
    result.context.precision = context.kv_quantization.bits;
  return result;
}
export function loadModelDrafts(storage: () => StorageLike): Record<string, LoadOptions> {
  try {
    const values = JSON.parse(storage().getItem('tyloinfer.model-load-options.v1') || '{}');
    return Object.fromEntries(Object.entries(values).filter(([id]) => /^[a-f0-9]{32}$/.test(id)).map(([id, value]) => [id, normalizeLoadOptions(value)]));
  } catch { return {}; }
}
export function saveModelDrafts(storage: () => StorageLike, values: Record<string, LoadOptions>) {
  try { storage().setItem?.('tyloinfer.model-load-options.v1', JSON.stringify(values)); } catch { /* Current-page load options remain available without browser storage. */ }
}
export function nativeLoadRequest(model: any, options: LoadOptions): Record<string, any> {
  const request: Record<string, any> = { model: model.id, pin: options.pin };
  if (options.context.context_size != null) request.context_size = options.context.context_size;
  const { reserve_percent, precision, vram_preallocated } = options.context;
  request.context_management = { reserve_percent, precision, vram_preallocated };
  if (model.structured_decisions) request.jev_weights = options.context.jev_weights;
  if (options.devices.length) request.device_ids = options.devices;
  if (!options.pin && options.ttl != null) request.idle_ttl_seconds = options.ttl;
  return request;
}
