import { validCompanionIds } from './companion-options.js';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
export const terminalStates = new Set(['succeeded', 'failed', 'cancelled', 'interrupted']);
const states = new Set(['queued', 'running', 'cancelling', ...terminalStates]);
export const isOperationId = value => typeof value === 'string' && UUID.test(value);

export class ModelBridgeError extends Error {
  constructor(code, message, details = {}) { super(message); this.code = code; this.details = details; }
}
const invalid = () => new ModelBridgeError('tyloinfer/core-invalid-response', 'Invalid model operation response');
const unknown = operationId => new ModelBridgeError('tyloinfer/outcome-unknown', 'The write outcome is unknown', { operation_id: operationId, retryable: false });

export function createModelApi(connection) {
  const call = async (method, payload, signal, write = false) => {
    let reply;
    try { reply = await connection.rpc.call('/tyloinfer', method, payload, signal); }
    catch { throw write ? unknown(payload.operation_id) : new ModelBridgeError('tyloinfer/core-unavailable', 'Model bridge request did not complete'); }
    if (reply?.ok === false && typeof reply.error?.code === 'string') throw new ModelBridgeError(reply.error.code, reply.error.message, reply.error.details);
    if (reply?.ok !== true) throw write ? unknown(payload.operation_id) : invalid();
    return reply.value;
  };
  const job = (value, id, write = false) => {
    if (!value || value.id !== id || !['model.load', 'model.unload'].includes(value.kind) || !states.has(value.status) || typeof value.cancel_requested !== 'boolean') throw write ? unknown(id) : invalid();
    return value;
  };
  const submit = async (method, request, signal, companionIds) => {
    if (method === 'models/load' && (!request || Object.hasOwn(request, 'gguf_companions') || companionIds !== undefined && !validCompanionIds(companionIds))) throw new ModelBridgeError('tyloinfer/bad-request', 'Select registered companion identities; filesystem paths are not accepted');
    const value = await call(method, { request, ...(method === 'models/load' && companionIds?.length ? { companion_ids: [...companionIds] } : {}) }, signal, true);
    if (!isOperationId(value?.operation_id) || value.status !== 'accepted') throw unknown();
    return value;
  };
  return {
    load: (request, signal, companionIds) => submit('models/load', request, signal, companionIds),
    unload: (request, signal) => submit('models/unload', request, signal),
    get: async (id, signal) => job(await call('operations/get', { operation_id: id }, signal), id),
    cancel: async (id, signal) => job(await call('operations/cancel', { operation_id: id }, signal, true), id, true),
  };
}
