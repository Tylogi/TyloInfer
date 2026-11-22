import assert from 'node:assert/strict';
import { setTimeout } from 'node:timers/promises';
import { createCoreBridge } from '../core-bridge.js';

const [origin, artifact, failedDownload] = process.argv.slice(2);
assert.equal(new URL(origin).hostname, '127.0.0.1');
const instance = '11111111-1111-4111-8111-111111111111';
const source = { provider: 'huggingface', repo_id: 'example/model' };
const handler = createCoreBridge({ getProvider: () => ({ baseURL: origin + '/v1' }), resolveCredential: async () => undefined });
const call = (endpoint, payload) => handler(endpoint, payload, undefined, { principal: { kind: 'owner' } });
const value = async (endpoint, payload) => {
  const result = await call(endpoint, payload);
  assert.equal(result.ok, true, JSON.stringify(result));
  return result.value;
};
const waitFor = async (endpoint, operation_id, status) => {
  for (let attempt = 0; attempt < 300; attempt++) {
    const current = await value(endpoint, { operation_id });
    if (current.status === status) return current;
    assert.ok(!['failed', 'interrupted', 'cancelled'].includes(current.status));
    await setTimeout(10);
  }
  throw new Error('Controlled operation did not settle');
};
assert.deepEqual(await value('hub/readme', { ...source, revision: 'main' }), { content: '# Fixture model' });
assert.deepEqual(await value('hub/readme', { ...source, revision: 'missing' }), { content: null });
assert.deepEqual(await value('hub/probe-sources', { sources: [source] }), { data: [{ ...source, bytes_per_second: 0 }] });
const directory = (await value('core/read', { resource: 'download_directory' })).actual_directory;
for (const download_directory of [undefined, directory]) {
  const request = { ...source, destination: 'models/reuse', ...(download_directory === undefined ? {} : { download_directory }) };
  assert.deepEqual(await value('hub/download-reuse', { request }), { reusable_files: { 'fixture.tq': 7 } });
}
const policy = await value('instances/configure', { instance_id: instance, request: { pinned: true, idle_ttl_seconds: null }, confirmed: true });
assert.equal(policy.id, instance);
assert.equal(policy.pinned, true);
assert.equal(policy.idle_ttl_seconds, null);
assert.equal((await value('instances/yarn', { instance_id: instance })).supported, false);
const configured = await value('context/configure', { request: { instance_id: instance, context_size: 1024, yarn_enabled: false,
  kv_memory: { qsa_mode: 'auto' } }, confirmed: true });
assert.equal(configured.kind, 'runtime.context.configure');
const refused = await fetch(origin + '/api/v1/jobs/' + configured.id + '/cancel', { method: 'POST', redirect: 'error' });
assert.equal(refused.status, 409);
assert.equal((await refused.json()).error.code, 'context_reload_in_progress');
assert.equal((await call('context/cancel', { operation_id: configured.id, confirmed: true })).error.code, 'tyloinfer/bad-request');
assert.equal((await fetch(origin + '/fixture/release-context', { method: 'POST', redirect: 'error' })).status, 200);
assert.equal((await waitFor('context/get', configured.id, 'succeeded')).result.fixture, true);
const deletion = await value('models/deletion-plan', { model_id: artifact });
assert.equal(deletion.model_id, artifact);
assert.equal(deletion.files.length, 1);
assert.ok(deletion.files.every(file => /[\\/]deletable\.tq$/.test(file)));
assert.ok(deletion.total_bytes > 0);
const stale = await call('models/delete', { model_id: artifact, token: '0'.repeat(64), confirmed: true });
assert.equal(stale.error.code, 'tyloinfer/core-rejected');
assert.equal(stale.error.details.http_status, 409);
const deleted = await value('models/delete', { model_id: artifact, token: deletion.token, confirmed: true });
assert.deepEqual(deleted, { files: 1, total_bytes: deletion.total_bytes });
const retried = await value('downloads/retry', { operation_id: failedDownload, confirmed: true });
assert.equal(retried.kind, 'download.huggingface');
await waitFor('downloads/get', retried.id, 'succeeded');
assert.equal((await call('context/get', { operation_id: retried.id })).error.code, 'tyloinfer/operation-not-supported');
assert.equal(await value('downloads/delete', { operation_id: retried.id, confirmed: true }), null);
const archived = await call('downloads/get', { operation_id: retried.id });
assert.equal(archived.error.details.http_status, 404);
assert.equal((await value('core/read', { resource: 'instances' })).data.length, 0);
console.log('CORE_MODEL_ACTIONS_BRIDGE_API_PASS; FILES=isolated-fixtures; RUNTIME_HUB_DOWNLOAD=controlled; REAL_MODELS=none');
