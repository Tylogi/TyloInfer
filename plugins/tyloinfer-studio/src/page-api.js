import { ModelBridgeError } from './model-api.js';
import { validateConnectionInfo } from './connection-info.js';

export const pageResources = {
  overview: ['health', 'status', 'serving', 'instances', 'jobs'],
  models: ['status', 'instances', 'jobs'],
  service: ['listener', 'memory_policy', 'inference_policy', 'context_policy', 'model_aliases', 'authentication_policy', 'status', 'jobs'],
  resources: ['status', 'resources', 'prefix_cache', 'instances'],
  downloads: ['download_directory', 'jobs', 'job_kinds'],
  logs: ['logs', 'metrics', 'jobs'],
};

export function createPageApi(connection) {
  const call = async (method, payload, signal, write = false, allowNull = false) => {
    let reply;
    try { reply = await connection.rpc.call('/tyloinfer', method, payload, signal); }
    catch { throw new ModelBridgeError(write ? 'tyloinfer/outcome-unknown' : 'tyloinfer/core-unavailable', 'The core request did not complete', { retryable: false }); }
    if (reply?.ok === false && typeof reply.error?.code === 'string' && typeof reply.error?.message === 'string') throw new ModelBridgeError(reply.error.code, reply.error.message, reply.error.details);
    if (reply?.ok !== true || reply.value === undefined || (!allowNull && reply.value === null)) throw new ModelBridgeError(write ? 'tyloinfer/outcome-unknown' : 'tyloinfer/core-invalid-response', 'Invalid core response', { retryable: false });
    return reply.value;
  };
  const read = (resource, params, signal) => call('core/read', { resource, ...(params && Object.keys(params).length ? { params } : {}) }, signal);
  return {
    read,
    connectionInfo: async signal => validateConnectionInfo(await call('connection/info', {}, signal)),
    page: async (view, params = {}, signal) => Object.fromEntries(await Promise.all((pageResources[view] || []).map(async resource => {
      try { return [resource, { value: await read(resource, params[resource], signal) }]; }
      catch (error) { return [resource, { error }]; }
    }))),
    analysis: (modelId, signal) => call('models/analysis', { model_id: modelId }, signal),
    cacheProfile: (modelId, includeRecurrent, signal) => call('models/cache-profile', { model_id: modelId, ...(includeRecurrent === undefined ? {} : { include_recurrent: includeRecurrent }) }, signal, false, true),
    loadPlan: (modelId, params, signal) => call('models/load-plan', { model_id: modelId, ...params }, signal),
    directory: (modelId, signal) => call('models/directory', { model_id: modelId }, signal),
    browseDirectory: (path, signal) => call('models/browse-directory', { path }, signal),
    openDirectory: (directoryId, signal) => call('models/open-directory', { directory_id: directoryId }, signal, true),
    remoteNodes: async (refresh, signal) => (await call('service/remote-nodes', { refresh }, signal)).data,
    configureListener: (protocol, port, signal) => call('service/configure-listener', { protocol, port }, signal, true),
    configureAuthentication: (enabled, apiKey, signal) => call('service/configure-authentication',
      { enabled, ...(apiKey ? { api_key: apiKey } : {}) }, signal, true),
    addRemoteNode: (request, signal) => call('service/add-remote-node', request, signal, true),
    deleteRemoteNode: (id, signal) => call('service/delete-remote-node', { id }, signal, true, true),
    job: (jobId, signal) => call('jobs/get', { job_id: jobId }, signal),
    deleteJob: (jobId, signal) => call('jobs/delete', { job_id: jobId, confirmed: true }, signal, true, true),
    clearCompletedJobs: signal => call('jobs/clear-completed', { scope: 'all-completed', confirmed: true }, signal, true, true),
    jobEvents: (jobId, after, signal) => call('jobs/events', { job_id: jobId, after, limit: 200, tail: false }, signal),
    configureInstance: (instanceId, request, signal) => call('instances/configure', { instance_id: instanceId, request, confirmed: true }, signal, true),
    yarn: (instanceId, signal) => call('instances/yarn', { instance_id: instanceId }, signal),
    configureContext: (request, signal) => call('context/configure', { request, confirmed: true }, signal, true),
    contextGet: (operationId, signal) => call('context/get', { operation_id: operationId }, signal),
    configureCacheDirectory: (directoryId, signal) => call('cache/configure-directory', { directory_id: directoryId, scope: 'all-loaded-models', confirmed: true }, signal, true),
    clearCache: (instanceId, signal) => call('cache/clear', { instance_id: instanceId, confirmed: true }, signal, true),
    trimCache: (instanceId, targetBytes, signal) => call('cache/trim', { instance_id: instanceId, target_bytes: targetBytes, confirmed: true }, signal, true),
    cacheText: (namespace, block, offset, signal) => call('cache/text', { namespace, block, offset, limit: 8192 }, signal),
    purgeCache: (target, signal) => call('cache/purge', { ...target, confirmed: true }, signal, true),
    deletionPlan: (modelId, signal) => call('models/deletion-plan', { model_id: modelId }, signal),
    deleteArtifact: (modelId, token, signal) => call('models/delete', { model_id: modelId, token, confirmed: true }, signal, true),
    register: (directoryId, signal, fileName) => call('models/register', { directory_id: directoryId, ...(fileName === undefined ? {} : { file_name: fileName }) }, signal, true),
    configure: (resource, request, signal) => call('service/configure', { resource, request }, signal, true),
    serviceGet: (operationId, signal) => call('service/get', { operation_id: operationId }, signal),
    serviceCancel: (operationId, signal) => call('service/cancel', { operation_id: operationId }, signal, true),
    search: (request, signal) => call('hub/search', request, signal),
    info: (request, signal) => call('hub/info', request, signal),
    official: (request, signal) => call('hub/official', request, signal),
    officialLocal: signal => call('hub/official-local', {}, signal),
    readme: (request, signal) => call('hub/readme', request, signal),
    probeSources: (sources, signal) => call('hub/probe-sources', { sources }, signal),
    reuse: (request, signal) => call('hub/download-reuse', { request }, signal),
    configureDownloadDirectory: (request, signal) => call('downloads/configure-directory', { request, confirmed: true }, signal, true),
    download: (provider, request, signal) => call('downloads/create', { provider, request }, signal, true),
    downloadRetry: (operationId, signal) => call('downloads/retry', { operation_id: operationId, confirmed: true }, signal, true),
    downloadDelete: (operationId, signal) => call('downloads/delete', { operation_id: operationId, confirmed: true }, signal, true, true),
    downloadGet: (operationId, signal) => call('downloads/get', { operation_id: operationId }, signal),
    downloadCancel: (operationId, signal) => call('downloads/cancel', { operation_id: operationId }, signal, true),
  };
}
