import { validCompanionIds, validCompanionPaths } from './companion-options.ts';
import type { RpcConnection, RpcPayload } from './types.ts';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
export const terminalStates = new Set(['succeeded', 'failed', 'cancelled', 'interrupted']);
const states = new Set(['queued', 'running', 'cancelling', ...terminalStates]);
export const isOperationId = (value: unknown): value is string => typeof value === 'string' && UUID.test(value);

export class ModelBridgeError extends Error {
  code: string;
  details: Record<string, any>;
  constructor(code: string, message: string, details: Record<string, any> = {}) { super(message); this.code = code; this.details = details; }
}
const invalid = () => new ModelBridgeError('tyloinfer/core-invalid-response', 'Invalid model operation response');
const unknown = (operationId?: string | null) => new ModelBridgeError('tyloinfer/outcome-unknown', 'The write outcome is unknown', { operation_id: operationId, retryable: false });

export function createModelApi(connection: RpcConnection) {
  const call = async (method: string, payload: RpcPayload, signal?: AbortSignal, write = false) => {
    let reply;
    try { reply = await connection.rpc.call('/tyloinfer', method, payload, signal); }
    catch { throw write ? unknown(payload.operation_id) : new ModelBridgeError('tyloinfer/core-unavailable', 'Model bridge request did not complete'); }
    if (reply?.ok === false && typeof reply.error?.code === 'string') throw new ModelBridgeError(reply.error.code, reply.error.message, reply.error.details);
    if (reply?.ok !== true) throw write ? unknown(payload.operation_id) : invalid();
    return reply.value;
  };
  const job = (value: any, id: string, write = false) => {
    if (!value || value.id !== id || !['model.load', 'model.unload'].includes(value.kind) || !states.has(value.status) || typeof value.cancel_requested !== 'boolean') throw write ? unknown(id) : invalid();
    return value;
  };
  const submit = async (method: 'models/load' | 'models/unload', request: RpcPayload, signal?: AbortSignal, companions?: string[]) => {
    let companionFields: RpcPayload = {};
    if (method === 'models/load') {
      if (!request || Object.hasOwn(request, 'gguf_companions')) throw new ModelBridgeError('tyloinfer/bad-request', 'Use the Engram companion picker');
      if (companions !== undefined) {
        if (validCompanionPaths(companions)) companionFields = { companion_paths: [...companions] };
        else if (validCompanionIds(companions)) companionFields = companions.length ? { companion_ids: [...companions] } : {};
        else throw new ModelBridgeError('tyloinfer/bad-request', 'Choose registered companions or enter full server paths');
      }
    }
    const value = await call(method, { request, ...companionFields }, signal, true);
    if (!isOperationId(value?.operation_id) || value.status !== 'accepted') throw unknown();
    return value;
  };
  return {
    load: (request: RpcPayload, signal?: AbortSignal, companionIds?: string[]) => submit('models/load', request, signal, companionIds),
    unload: (request: RpcPayload, signal?: AbortSignal) => submit('models/unload', request, signal),
    get: async (id: string, signal?: AbortSignal) => job(await call('operations/get', { operation_id: id }, signal), id),
    cancel: async (id: string, signal?: AbortSignal) => job(await call('operations/cancel', { operation_id: id }, signal, true), id, true),
  };
}
