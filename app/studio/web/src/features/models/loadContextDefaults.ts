import { getApiBaseUrl } from '../../shared/api/client';
import type { LoadContextDefaults } from '@shared/types/load-context';

export type { LoadContextDefaults };

export const DEFAULT_LOAD_CONTEXT: LoadContextDefaults = {
  context_size: null, reserve_percent: 10, precision: null, jev_weights: 'stream', vram_preallocated: true,
};

function key(model?: string) {
  return `tyloinfer.studio.load-context.v1:${getApiBaseUrl() || window.location.origin}${model ? `:model:${model}` : ''}`;
}

export function readLoadContext(model?: string): LoadContextDefaults {
  try {
    const value = JSON.parse(localStorage.getItem(key(model)) || '{}');
    if (!value || typeof value !== 'object' || Array.isArray(value)) return structuredClone(DEFAULT_LOAD_CONTEXT);
    const result = structuredClone(DEFAULT_LOAD_CONTEXT);
    if (Number.isSafeInteger(value.context_size) && value.context_size >= 512 && value.context_size <= 2147483647)
      result.context_size = value.context_size;
    if ([5, 10, 20].includes(value.reserve_percent)) result.reserve_percent = value.reserve_percent;
    if ([2, 2.5, 3, 3.5, 4, 6, 8].includes(value.precision)) result.precision = value.precision;
    if (typeof value.vram_preallocated === 'boolean' && (Object.hasOwn(value, 'reserve_percent') || value.automatic === false))
      result.vram_preallocated = value.vram_preallocated;
    if (!Object.hasOwn(value, 'precision') && value.automatic === false && value.kv_quantization?.enabled === true
        && [2, 2.5, 3, 3.5, 4, 6, 8].includes(value.kv_quantization.bits))
      result.precision = value.kv_quantization.bits;
    if (value.jev_weights === 'stream' || value.jev_weights === 'vram') result.jev_weights = value.jev_weights;
    return result;
  } catch { return structuredClone(DEFAULT_LOAD_CONTEXT); }
}

export function saveLoadContext(value: LoadContextDefaults, model?: string) {
  try { localStorage.setItem(key(model), JSON.stringify(value)); } catch { /* The current page remains usable without storage. */ }
}
