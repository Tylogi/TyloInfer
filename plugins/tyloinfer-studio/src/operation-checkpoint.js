import { isOperationId } from './model-api.js';

const kinds = new Set(['model.load', 'model.unload', 'memory', 'download', 'context']);
export function createOperationCheckpoint(storage, key) {
  const source = () => typeof storage === 'function' ? storage() : storage;
  return {
    load() {
      try {
        const value = JSON.parse(source().getItem(key) || 'null');
        if (!value || value.version !== 1 || (value.id !== null && !isOperationId(value.id)) || (value.kind !== null && !kinds.has(value.kind)) || typeof value.unknown !== 'boolean') return null;
        return { id: value.id, kind: value.kind, unknown: value.unknown, label: typeof value.label === 'string' ? value.label.slice(0, 500) : '' };
      } catch { return null; }
    },
    save(value) {
      try {
        if (!value) source().removeItem(key);
        else source().setItem(key, JSON.stringify({ version: 1, id: value.id || null, kind: value.kind || null, unknown: value.unknown === true, label: value.label || '' }));
      } catch { /* Tracking remains available for the current page when browser storage is unavailable. */ }
    },
  };
}
