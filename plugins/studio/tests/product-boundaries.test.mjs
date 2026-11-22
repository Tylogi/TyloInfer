import test from 'node:test';
import assert from 'node:assert/strict';
import { createPageApi } from '../src/page-api.js';
import { createModelApi } from '../src/model-api.js';
import { createModelManager } from '../src/model-manager.js';
import { validateConnectionInfo } from '../src/connection-info.js';
import { companionsAvailable, registeredCompanions, validCompanionIds } from '../src/companion-options.js';
const id = '11111111-1111-4111-8111-111111111111';
const main = 'a'.repeat(32), companion = 'b'.repeat(32), folder = 'c'.repeat(32);
const metadata = { base_url: 'https://core.example/prefix', openai_base_url: 'https://core.example/prefix/v1', anthropic_base_url: null, anthropic_port: null, protocol_version: '1.0' };
const ok = value => ({ ok: true, value });

test('uses only validated provider endpoint metadata without an Agent-origin fallback', async () => {
  const calls = [], controller = new AbortController();
  const api = createPageApi({ rpc: { call: async (...args) => { calls.push(args); return ok(metadata); } } });
  assert.deepEqual(await api.connectionInfo(controller.signal), metadata);
  assert.deepEqual(calls[0], ['/tyloinfer', 'connection/info', {}, controller.signal]);
  assert.equal(validateConnectionInfo({ ...metadata, anthropic_port: 9443, anthropic_base_url: 'https://core.example:9443/prefix/v1' }).anthropic_port, 9443);
});

test('rejects credentials, query fragments, invalid listeners and unrelated endpoint metadata', () => {
  for (const value of [null, { ...metadata, base_url: 'https://user:password@core.example/prefix' }, { ...metadata, base_url: 'https://core.example/prefix?token=example' }, { ...metadata, openai_base_url: 'https://other.example/v1' }, { ...metadata, openai_base_url: 'https://core.example/prefix/v1#secret' }, { ...metadata, anthropic_port: 0 }, { ...metadata, anthropic_port: 9443, anthropic_base_url: null }, { ...metadata, anthropic_port: null, anthropic_base_url: 'https://core.example/v1' }, { ...metadata, protocol_version: 'unknown' }]) assert.throws(() => validateConnectionInfo(value), error => error.code === 'tyloinfer/core-invalid-response');
});

test('preserves native null archive success with explicit single and all-completed scopes', async () => {
  const calls = [];
  const api = createPageApi({ rpc: { call: async (...args) => { calls.push(args); return ok(null); } } });
  assert.equal(await api.deleteJob(id), null); assert.equal(await api.clearCompletedJobs(), null);
  assert.deepEqual(calls.map(call => call.slice(0, 3)), [['/tyloinfer', 'jobs/delete', { job_id: id, confirmed: true }], ['/tyloinfer', 'jobs/clear-completed', { scope: 'all-completed', confirmed: true }]]);
});

test('submits a cache directory identity with explicit global scope and preserves the native memory ticket', async () => {
  const calls = [], ticket = { status: 'accepted', operation_id: id };
  const api = createPageApi({ rpc: { call: async (...args) => { calls.push(args); return ok(ticket); } } });
  assert.deepEqual(await api.configureCacheDirectory(folder), ticket);
  assert.deepEqual(calls[0].slice(0, 3), ['/tyloinfer', 'cache/configure-directory', { directory_id: folder, scope: 'all-loaded-models', confirmed: true }]);
});

test('never retries archival or global directory writes after unknown outcomes', async () => {
  for (const invoke of [api => api.deleteJob(id), api => api.clearCompletedJobs(), api => api.configureCacheDirectory(folder)]) {
    let calls = 0; const api = createPageApi({ rpc: { call: async () => { calls++; throw new Error('disconnected'); } } });
    await assert.rejects(invoke(api), error => error.code === 'tyloinfer/outcome-unknown'); assert.equal(calls, 1);
  }
});

test('passes registered companion identities outside the native request and rejects relative paths', async () => {
  const calls = [], controller = new AbortController();
  const api = createModelApi({ rpc: { call: async (...args) => { calls.push(args); return ok({ status: 'accepted', operation_id: id }); } } });
  await api.load({ model: main, pin: false }, controller.signal, [companion]);
  assert.deepEqual(calls[0], ['/tyloinfer', 'models/load', { request: { model: main, pin: false }, companion_ids: [companion] }, controller.signal]);
  for (const ids of [['arbitrary/path'], [companion, companion], Array.from({ length: 17 }, (_, index) => index.toString(16).padStart(32, '0'))]) await assert.rejects(api.load({ model: main }, undefined, ids), error => error.code === 'tyloinfer/bad-request');
  await assert.rejects(api.load({ model: main, gguf_companions: ['arbitrary/path'] }), error => error.code === 'tyloinfer/bad-request');
  assert.equal(calls.length, 1);
});
test('loads a model with an explicitly supplied absolute companion path', async () => {
  const writes = [];
  const model = { id: main, name: 'Main', format: 'gguf', complete: true, loadable: true, gguf_companion_required: true };
  const connection = { rpc: { call: async (_channel, method, payload) => {
    if (method === 'core/read') return ok({ data: payload.resource === 'models' ? [model] : [] });
    writes.push({ method, payload }); return ok({ status: 'accepted', operation_id: id });
  } } };
  const manager = createModelManager(connection);
  try {
    const path = '/models/engram.gguf';
    await manager.refresh();
    assert.equal(await manager.load({ model: main }, [path]), true);
    assert.deepEqual(writes, [{ method: 'models/load', payload: { request: { model: main }, companion_paths: [path] } }]);
  } finally { manager.dispose(); }
});

test('offers only complete registered GGUF artifacts and rejects removed or incompatible identities', () => {
  const artifacts = [{ id: main, complete: true, format: 'gguf' }, { id: companion, complete: true, format: 'gguf', asset_kind: 'engram' }, { id: folder, complete: false, format: 'gguf' }, { id: 'd'.repeat(32), complete: true, format: 'tq' }];
  assert.deepEqual(registeredCompanions(artifacts, main).map(item => item.id), [companion]);
  assert.equal(companionsAvailable(artifacts, main, [companion]), true);
  assert.equal(companionsAvailable(artifacts, main, [folder]), false);
  assert.equal(validCompanionIds([companion, companion]), false);
});

test('rechecks required companion identity against current catalog before any load write', async () => {
  const writes = [];
  const models = [{ id: main, name: 'Main', format: 'gguf', complete: true, loadable: true, gguf_companion_required: true }, { id: companion, name: 'Table', format: 'gguf', complete: true, asset_kind: 'engram' }];
  const connection = { rpc: { call: async (_channel, method, payload) => {
    if (method === 'core/read') return ok({ data: payload.resource === 'models' ? models : [] });
    writes.push({ method, payload }); return ok({ status: 'accepted', operation_id: id });
  } } };
  const manager = createModelManager(connection);
  try {
    await manager.refresh(); assert.equal(await manager.load({ model: main }), false);
    assert.equal(await manager.load({ model: main }, [folder]), false); assert.equal(writes.length, 0);
    assert.equal(await manager.load({ model: main }, [companion]), true);
    assert.deepEqual(writes[0].payload, { request: { model: main }, companion_ids: [companion] });
  } finally { manager.dispose(); }
});
