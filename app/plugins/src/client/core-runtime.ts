import type { RpcConnection, RpcPayload } from './types.ts';

export async function describeCoreRuntime(connection: RpcConnection, signal?: AbortSignal) {
  const read = async (resource: string, params?: RpcPayload) => {
    const response = await connection.rpc.call('/tyloinfer', 'core/read', { resource, ...(params ? { params } : {}) }, signal);
    if (!response?.ok) throw new Error(response?.error?.message || 'Inference core unavailable');
    if (!Array.isArray(response.value?.data)) throw new Error('Invalid inference inventory response');
    return response.value.data;
  };
  const [models, loadedModels, instances] = await Promise.all([read('models', { include_components: true }), read('loaded_models'), read('instances')]);
  return { models, loadedModels, instances };
}
