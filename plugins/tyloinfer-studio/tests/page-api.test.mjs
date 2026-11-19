import test from 'node:test';
import assert from 'node:assert/strict';
import { createPageApi, pageResources } from '../src/page-api.js';
const ok = value => ({ ok: true, value });

test('reads page-specific inventories without automatic Hub or write calls', async () => {
  const calls = [];
  const controller = new AbortController();
  const api = createPageApi({ rpc: { call: async (...args) => { calls.push(args); return ok({ data: [] }); } } });
  for (const view of Object.keys(pageResources)) await api.page(view, { logs: { limit: 100 } }, controller.signal);
  assert.ok(calls.every(call => call[0] === '/tyloinfer' && call[1] === 'core/read' && call[3] === controller.signal));
  assert.deepEqual(calls.find(call => call[2].resource === 'logs')[2], { resource: 'logs', params: { limit: 100 } });
  assert.ok(pageResources.service.includes('context_policy'));
  assert.ok(pageResources.resources.includes('prefix_cache'));
  assert.ok(pageResources.downloads.includes('job_kinds'));
});
test('reads the local official snapshot with an empty payload and never falls back to network metadata', async () => {
  const calls = [];
  const controller = new AbortController();
  const api = createPageApi({ rpc: { call: async (...args) => { calls.push(args); return { ok: false, error: { code: 'tyloinfer/operation-not-supported', message: 'Local snapshot unavailable' } }; } } });
  await assert.rejects(api.officialLocal(controller.signal), error => error.code === 'tyloinfer/operation-not-supported');
  assert.deepEqual(calls.map(call => call.slice(0, 3)), [['/tyloinfer', 'hub/official-local', {}]]);
  assert.equal(calls[0][3], controller.signal);
});
test('keeps partial read failures visible rather than fabricating empty data', async () => {
  const api = createPageApi({ rpc: { call: async (_channel, _method, payload) => payload.resource === 'metrics' ? { ok: false, error: { code: 'tyloinfer/core-rejected', message: 'Denied', details: { http_status: 501 } } } : ok({ data: [] }) } });
  const value = await api.page('logs');
  assert.equal(value.metrics.error.code, 'tyloinfer/core-rejected');
  assert.equal(value.metrics.value, undefined);
  assert.deepEqual(value.logs.value, { data: [] });
});
test('forwards only named page actions and native download tickets without reshaping acceptance', async () => {
  const calls = [];
  const ticket = { id: '11111111-1111-4111-8111-111111111111', kind: 'download.huggingface', status: 'queued', cancel_requested: false };
  const api = createPageApi({ rpc: { call: async (...args) => { calls.push(args); return ok(ticket); } } });
  const request = { repo_id: 'example/model', destination: 'models/example/model', include: ['*.gguf'], expected_bytes: 1234, max_workers: 2 };
  assert.deepEqual(await api.download('huggingface', request), ticket);
  assert.deepEqual(calls[0].slice(0, 3), ['/tyloinfer', 'downloads/create', { provider: 'huggingface', request }]);
  await api.register('a'.repeat(32));
  assert.deepEqual(calls[1].slice(0, 3), ['/tyloinfer', 'models/register', { directory_id: 'a'.repeat(32) }]);
  await api.configure('inference_policy', { mtp_enabled: false });
  assert.deepEqual(calls[2][2], { resource: 'inference_policy', request: { mtp_enabled: false } });
});
test('does not retry page writes on transport failures or malformed response envelopes', async () => {
  for (const response of [null, { ok: false }, { ok: true }]) {
    let count = 0;
    const api = createPageApi({ rpc: { call: async () => { count++; if (response === null) throw new Error('disconnected'); return response; } } });
    await assert.rejects(api.configure('context_policy', { max_context_size: null }), error => error.code === 'tyloinfer/outcome-unknown');
    assert.equal(count, 1);
  }
});
test('preserves null cache metadata and forwards fixed model-detail reads', async () => {
  const calls = [];
  const api = createPageApi({ rpc: { call: async (...args) => { calls.push(args); return ok(null); } } });
  assert.equal(await api.cacheProfile('a'.repeat(32), undefined), null);
  assert.deepEqual(calls[0][2], { model_id: 'a'.repeat(32) });
  await api.cacheProfile('a'.repeat(32), true);
  assert.deepEqual(calls[1][2], { model_id: 'a'.repeat(32), include_recurrent: true });
  await assert.rejects(api.directory('a'.repeat(32)), error => error.code === 'tyloinfer/core-invalid-response');
  assert.equal(calls[2][1], 'models/directory');
});
test('reads selected job details and forward event pages without changing native events', async () => {
  const calls = [];
  const controller = new AbortController();
  const event = { job_id: '11111111-1111-4111-8111-111111111111', sequence: 201, type: 'progress', level: 'info', message: null, data: { progress: 0.5 }, created_at: '2026-01-01T00:00:00Z' };
  const api = createPageApi({ rpc: { call: async (...args) => { calls.push(args); return ok({ data: [event] }); } } });
  await api.job(event.job_id, controller.signal);
  assert.deepEqual(await api.jobEvents(event.job_id, 200, controller.signal), { data: [event] });
  assert.deepEqual(calls.map(call => call.slice(0, 3)), [
    ['/tyloinfer', 'jobs/get', { job_id: event.job_id }],
    ['/tyloinfer', 'jobs/events', { job_id: event.job_id, after: 200, limit: 200, tail: false }],
  ]);
  assert.ok(calls.every(call => call[3] === controller.signal));
});
test('uses confirmed fixed model, cache and directory actions while preserving native null deletion', async () => {
  const calls = [];
  const controller = new AbortController();
  const id = '11111111-1111-4111-8111-111111111111';
  const api = createPageApi({ rpc: { call: async (...args) => { calls.push(args); return ok(args[1] === 'downloads/delete' ? null : { status: 'ok' }); } } });
  await api.configureInstance(id, { pinned: false, idle_ttl_seconds: null }, controller.signal);
  await api.yarn(id, controller.signal);
  await api.configureContext({ instance_id: id, context_size: null }, controller.signal);
  await api.contextGet(id, controller.signal);
  await api.deletionPlan('a'.repeat(32), controller.signal);
  await api.deleteArtifact('a'.repeat(32), 'b'.repeat(64), controller.signal);
  await api.cacheText('c'.repeat(64), 'd'.repeat(64), 8192, controller.signal);
  await api.purgeCache({ scope: 'namespace', namespace: 'c'.repeat(64) }, controller.signal);
  await api.configureDownloadDirectory({ evict_prefix_cache: false }, controller.signal);
  await api.downloadRetry(id, controller.signal);
  assert.equal(await api.downloadDelete(id, controller.signal), null);
  assert.deepEqual(calls.map(call => call[1]), ['instances/configure', 'instances/yarn', 'context/configure', 'context/get', 'models/deletion-plan', 'models/delete', 'cache/text', 'cache/purge', 'downloads/configure-directory', 'downloads/retry', 'downloads/delete']);
  for (const index of [0, 2, 5, 7, 8, 9, 10]) assert.equal(calls[index][2].confirmed, true);
  assert.deepEqual(calls[2][2], { request: { instance_id: id, context_size: null }, confirmed: true });
  assert.deepEqual(calls[6][2], { namespace: 'c'.repeat(64), block: 'd'.repeat(64), offset: 8192, limit: 8192 });
  assert.deepEqual(calls[8][2], { request: { evict_prefix_cache: false }, confirmed: true });
  assert.ok(calls.every(call => call[3] === controller.signal));
});
test('keeps Hub metadata POSTs read-only and exposes no context cancellation', async () => {
  const calls = [];
  const api = createPageApi({ rpc: { call: async (...args) => { calls.push(args); throw new Error('disconnected'); } } });
  for (const read of [() => api.readme({ provider: 'huggingface', repo_id: 'example/model', revision: 'main' }), () => api.probeSources([{ provider: 'modelscope', repo_id: 'example/model' }]), () => api.reuse({ provider: 'huggingface', repo_id: 'example/model', destination: 'models/example/model' })]) await assert.rejects(read(), error => error.code === 'tyloinfer/core-unavailable');
  assert.deepEqual(calls.map(call => call[1]), ['hub/readme', 'hub/probe-sources', 'hub/download-reuse']);
  assert.equal(api.contextCancel, undefined);
});
test('keeps runtime clear and trim fixed to an explicitly confirmed instance and target', async () => {
  const calls = [];
  const controller = new AbortController();
  const id = '11111111-1111-4111-8111-111111111111';
  const api = createPageApi({ rpc: { call: async (...args) => { calls.push(args); return ok({ status: 'ok', released_bytes: 0 }); } } });
  await api.clearCache(id, controller.signal);
  await api.trimCache(id, 0, controller.signal);
  assert.deepEqual(calls.map(call => call.slice(0, 3)), [
    ['/tyloinfer', 'cache/clear', { instance_id: id, confirmed: true }],
    ['/tyloinfer', 'cache/trim', { instance_id: id, target_bytes: 0, confirmed: true }],
  ]);
  assert.ok(calls.every(call => call[3] === controller.signal));
  let writes = 0;
  const disconnected = createPageApi({ rpc: { call: async () => { writes++; throw new Error('disconnected'); } } });
  await assert.rejects(disconnected.trimCache(id, 0), error => error.code === 'tyloinfer/outcome-unknown');
  assert.equal(writes, 1);
});

test('sends analysis and user-initiated metadata requests through fixed methods', async () => {
  const calls = [];
  const api = createPageApi({ rpc: { call: async (...args) => { calls.push(args); return ok({ data: [] }); } } });
  await api.analysis('a'.repeat(32));
  await api.search({ provider: 'modelscope', query: 'example' });
  await api.info({ provider: 'modelscope', repo_id: 'example/model' });
  await api.official({ refresh: false });
  assert.deepEqual(calls.map(call => call[1]), ['models/analysis', 'hub/search', 'hub/info', 'hub/official']);
  assert.deepEqual(calls[0][2], { model_id: 'a'.repeat(32) });
});
