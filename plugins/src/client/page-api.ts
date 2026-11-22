import { ModelBridgeError } from './model-api.ts';
import { validateConnectionInfo, type ConnectionInfo } from './connection-info.ts';
import type { RpcConnection, RpcPayload } from './types.ts';

export const pageResources = {
  overview: ['health', 'status', 'serving', 'instances', 'jobs'],
  models: ['status', 'instances', 'jobs'],
  service: ['listener', 'memory_policy', 'inference_policy', 'context_policy', 'model_aliases', 'authentication_policy', 'status', 'jobs'],
  resources: ['status', 'resources', 'prefix_cache', 'instances'],
  downloads: ['download_directory', 'jobs', 'job_kinds'],
  logs: ['logs', 'metrics', 'jobs'],
};

export type PageView = keyof typeof pageResources;

export function createPageApi(connection: RpcConnection) {
  const call = async (method: string, payload: RpcPayload, signal?: AbortSignal, write = false, allowNull = false) => {
    let reply;
    try { reply = await connection.rpc.call('/tyloinfer', method, payload, signal); }
    catch { throw new ModelBridgeError(write ? 'tyloinfer/outcome-unknown' : 'tyloinfer/core-unavailable', 'The core request did not complete', { retryable: false }); }
    if (reply?.ok === false && typeof reply.error?.code === 'string' && typeof reply.error?.message === 'string') throw new ModelBridgeError(reply.error.code, reply.error.message, reply.error.details);
    if (reply?.ok !== true || reply.value === undefined || (!allowNull && reply.value === null)) throw new ModelBridgeError(write ? 'tyloinfer/outcome-unknown' : 'tyloinfer/core-invalid-response', 'Invalid core response', { retryable: false });
    return reply.value;
  };
  const read = (resource: string, params?: RpcPayload, signal?: AbortSignal) => call('core/read', { resource, ...(params && Object.keys(params).length ? { params } : {}) }, signal);
  return {
    read,
    connectionInfo: async (signal?: AbortSignal): Promise<ConnectionInfo> => validateConnectionInfo(await call('connection/info', {}, signal)),
    page: async (view: PageView, params: RpcPayload = {}, signal?: AbortSignal) => Object.fromEntries(await Promise.all((pageResources[view] || []).map(async resource => {
      try { return [resource, { value: await read(resource, params[resource], signal) }]; }
      catch (error) { return [resource, { error }]; }
    }))),
    analysis: (modelId: string, signal?: AbortSignal) => call('models/analysis', { model_id: modelId }, signal),
    cacheProfile: (modelId: string, includeRecurrent?: boolean, signal?: AbortSignal) => call('models/cache-profile', { model_id: modelId, ...(includeRecurrent === undefined ? {} : { include_recurrent: includeRecurrent }) }, signal, false, true),
    loadPlan: (modelId: string, params: RpcPayload, signal?: AbortSignal) => call('models/load-plan', { model_id: modelId, ...params }, signal),
    directory: (modelId: string, signal?: AbortSignal) => call('models/directory', { model_id: modelId }, signal),
    browseDirectory: (path: string, signal?: AbortSignal) => call('models/browse-directory', { path }, signal),
    openDirectory: (directoryId: string, signal?: AbortSignal) => call('models/open-directory', { directory_id: directoryId }, signal, true),
    remoteNodes: async (refresh?: boolean, signal?: AbortSignal) => (await call('service/remote-nodes', { refresh }, signal)).data,
    configureListener: (protocol: string, port: number, signal?: AbortSignal) => call('service/configure-listener', { protocol, port }, signal, true),
    configureAuthentication: (enabled: boolean, apiKey?: string, signal?: AbortSignal) => call('service/configure-authentication',
      { enabled, ...(apiKey ? { api_key: apiKey } : {}) }, signal, true),
    addRemoteNode: (request: RpcPayload, signal?: AbortSignal) => call('service/add-remote-node', request, signal, true),
    deleteRemoteNode: (id: string, signal?: AbortSignal) => call('service/delete-remote-node', { id }, signal, true, true),
    job: (jobId: string, signal?: AbortSignal) => call('jobs/get', { job_id: jobId }, signal),
    deleteJob: (jobId: string, signal?: AbortSignal) => call('jobs/delete', { job_id: jobId, confirmed: true }, signal, true, true),
    clearCompletedJobs: (signal?: AbortSignal) => call('jobs/clear-completed', { scope: 'all-completed', confirmed: true }, signal, true, true),
    jobEvents: (jobId: string, after: number, signal?: AbortSignal) => call('jobs/events', { job_id: jobId, after, limit: 200, tail: false }, signal),
    configureInstance: (instanceId: string, request: RpcPayload, signal?: AbortSignal) => call('instances/configure', { instance_id: instanceId, request, confirmed: true }, signal, true),
    yarn: (instanceId: string, signal?: AbortSignal) => call('instances/yarn', { instance_id: instanceId }, signal),
    configureContext: (request: RpcPayload, signal?: AbortSignal) => call('context/configure', { request, confirmed: true }, signal, true),
    contextGet: (operationId: string, signal?: AbortSignal) => call('context/get', { operation_id: operationId }, signal),
    configureCacheDirectory: (directoryId: string, signal?: AbortSignal) => call('cache/configure-directory', { directory_id: directoryId, scope: 'all-loaded-models', confirmed: true }, signal, true),
    clearCache: (instanceId: string, signal?: AbortSignal) => call('cache/clear', { instance_id: instanceId, confirmed: true }, signal, true),
    trimCache: (instanceId: string, targetBytes: number, signal?: AbortSignal) => call('cache/trim', { instance_id: instanceId, target_bytes: targetBytes, confirmed: true }, signal, true),
    cacheText: (namespace: string, block: string, offset: number, signal?: AbortSignal) => call('cache/text', { namespace, block, offset, limit: 8192 }, signal),
    purgeCache: (target: RpcPayload, signal?: AbortSignal) => call('cache/purge', { ...target, confirmed: true }, signal, true),
    deletionPlan: (modelId: string, signal?: AbortSignal) => call('models/deletion-plan', { model_id: modelId }, signal),
    deleteArtifact: (modelId: string, token: string, signal?: AbortSignal) => call('models/delete', { model_id: modelId, token, confirmed: true }, signal, true),
    register: (directoryId: string, signal?: AbortSignal, fileName?: string) => call('models/register', { directory_id: directoryId, ...(fileName === undefined ? {} : { file_name: fileName }) }, signal, true),
    configure: (resource: string, request: RpcPayload, signal?: AbortSignal) => call('service/configure', { resource, request }, signal, true),
    serviceGet: (operationId: string, signal?: AbortSignal) => call('service/get', { operation_id: operationId }, signal),
    serviceCancel: (operationId: string, signal?: AbortSignal) => call('service/cancel', { operation_id: operationId }, signal, true),
    search: (request: RpcPayload, signal?: AbortSignal) => call('hub/search', request, signal),
    info: (request: RpcPayload, signal?: AbortSignal) => call('hub/info', request, signal),
    official: (request: RpcPayload, signal?: AbortSignal) => call('hub/official', request, signal),
    officialLocal: (signal?: AbortSignal) => call('hub/official-local', {}, signal),
    readme: (request: RpcPayload, signal?: AbortSignal) => call('hub/readme', request, signal),
    probeSources: (sources: RpcPayload[], signal?: AbortSignal) => call('hub/probe-sources', { sources }, signal),
    reuse: (request: RpcPayload, signal?: AbortSignal) => call('hub/download-reuse', { request }, signal),
    configureDownloadDirectory: (request: RpcPayload, signal?: AbortSignal) => call('downloads/configure-directory', { request, confirmed: true }, signal, true),
    download: (provider: string, request: RpcPayload, signal?: AbortSignal) => call('downloads/create', { provider, request }, signal, true),
    downloadRetry: (operationId: string, signal?: AbortSignal) => call('downloads/retry', { operation_id: operationId, confirmed: true }, signal, true),
    downloadDelete: (operationId: string, signal?: AbortSignal) => call('downloads/delete', { operation_id: operationId, confirmed: true }, signal, true, true),
    downloadGet: (operationId: string, signal?: AbortSignal) => call('downloads/get', { operation_id: operationId }, signal),
    downloadCancel: (operationId: string, signal?: AbortSignal) => call('downloads/cancel', { operation_id: operationId }, signal, true),
  };
}
