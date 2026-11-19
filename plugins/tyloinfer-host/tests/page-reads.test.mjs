import assert from 'node:assert/strict';
import test from 'node:test';
import { createCoreBridge } from '../core-bridge.js';

const owner = { principal: { kind: 'owner' } };
const instance = '11111111-1111-4111-8111-111111111111';
const artifact = '1234567890abcdef1234567890abcdef';
const health = { service: 'tyloinfer-server', protocol_version: '1.0' };
function harness(route = url => Response.json({ path: url.pathname, query: Object.fromEntries(url.searchParams) })) {
  const calls = [];
  const bridge = createCoreBridge({
    getProvider: () => ({ baseURL: 'http://core.invalid/prefix/v1' }),
    resolveCredential: async () => undefined,
    fetchImpl: async (url, init) => {
      calls.push({ url, ...init });
      return url.pathname.endsWith('/health') ? Response.json(health) : route(url, init);
    },
  });
  return { calls, call: (method, payload, peer = owner) => bridge(method, payload, undefined, peer) };
}
const resources = {
  listener: '/runtime/listener', memory_policy: '/runtime/memory-policy', inference_policy: '/runtime/inference-policy',
  context_policy: '/runtime/context-policy', model_aliases: '/runtime/model-aliases', authentication_policy: '/auth/policy',
  serving: '/runtime/serving', metrics: '/runtime/metrics', logs: '/runtime/logs', jobs: '/jobs', job_kinds: '/jobs/kinds',
  download_directory: '/hub/download-directory', model_directories: '/models/directories', prefix_cache: '/runtime/cache/entries',
};
for (const [resource, path] of Object.entries(resources)) {
  test('reads the fixed page resource without changing the configured prefix: ' + resource, async () => {
    const h = harness();
    const reply = await h.call('core/read', { resource });
    assert.deepEqual(reply, { ok: true, value: { path: '/prefix/api/v1' + path, query: {} } });
    assert.equal(h.calls.length, 2);
    assert.ok(h.calls.every(call => call.method === 'GET' && call.redirect === 'error'));
  });
  test('page resource requires owner before resolving or fetching: ' + resource, async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call('core/read', { resource }, { principal: { kind: 'viewer' } })).error.code, 'tyloinfer/forbidden');
    assert.equal(h.calls.length, 0);
  });
}
for (const [resource, params] of [
  ['status', { instance_id: instance }], ['capabilities', { instance_id: instance }],
  ['models', { refresh: true, include_components: false }], ['serving', { scope: 'alltime', model: 'model & x' }],
  ['metrics', { instance_id: instance, since: '2026-01-01T00:00:00Z', limit: 2000 }],
  ['logs', { instance_id: instance, level: 'warning', after: 0, limit: 1 }],
  ['jobs', { kind: 'download.huggingface', status: 'running', limit: 200, offset: 3 }],
  ['model_directories', { directory_id: artifact }],
  ['prefix_cache', { namespace: 'a'.repeat(64), offset: 0, limit: 200, query: 'name & value', search_in: 'id', text_filter: 'saved', chain_filter: 'complete', sort: 'size' }],
]) {
  test('serializes only named filters for ' + resource, async () => {
    const h = harness();
    const reply = await h.call('core/read', { resource, params });
    assert.equal(reply.ok, true);
    assert.deepEqual(reply.value.query, Object.fromEntries(Object.entries(params).map(([key, value]) => [key, String(value)])));
    assert.equal(h.calls.at(-1).url.hash, '');
    assert.equal(h.calls.filter(call => call.method !== 'GET').length, 0);
  });
}
for (const [resource, params] of [
  ['logs', { url: 'http://untrusted.invalid/' }], ['logs', { path: '/jobs' }], ['logs', { method: 'POST' }],
  ['logs', { limit: 2001 }], ['logs', { limit: 0 }], ['logs', { limit: '10' }], ['logs', { after: -1 }],
  ['logs', { level: 'trace' }], ['logs', { instance_id: '../outside' }],
  ['metrics', { since: 'yesterday' }], ['metrics', { since: '2026-01-01T00:00:00' }],
  ['serving', { scope: 'daily' }], ['serving', { model: 'bad\nvalue' }],
  ['jobs', { kind: '../jobs' }], ['jobs', { status: 'accepted' }], ['jobs', { limit: 201 }], ['jobs', { offset: 1.5 }],
  ['model_directories', { path: '/outside' }], ['model_directories', { directory_id: '..' }],
  ['prefix_cache', { namespace: 'a'.repeat(63) }], ['prefix_cache', { query: 'a'.repeat(201) }], ['prefix_cache', { sort: 'invalid' }],
  ['resources', { refresh: true }], ['models', { refresh: 'true' }], ['health', { instance_id: instance }],
  ['__proto__', {}], ['constructor', {}], ['logs', null], ['logs', []],
]) {
  test('rejects unsupported filters before network: ' + resource + ' ' + JSON.stringify(params), async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call('core/read', { resource, params })).error.code, 'tyloinfer/bad-request');
    assert.equal(h.calls.length, 0);
  });
}
test('model analysis uses the native artifact id and never a filesystem path', async () => {
  const h = harness();
  const reply = await h.call('models/analysis', { model_id: artifact });
  assert.equal(reply.ok, true);
  assert.equal(reply.value.path, '/prefix/api/v1/models/' + artifact + '/analysis');
  assert.equal(h.calls.filter(call => call.method !== 'GET').length, 0);
});
for (const payload of [{ model_id: '../outside' }, { model_id: instance }, { model_id: 'model-name' }, { model_id: artifact, path: '/outside' }, {}, null]) {
  test('rejects invalid analysis identity: ' + JSON.stringify(payload), async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call('models/analysis', payload)).error.code, 'tyloinfer/bad-request');
    assert.equal(h.calls.length, 0);
  });
}
test('analysis requires owner before any network access', async () => {
  const h = harness(() => assert.fail('network'));
  assert.equal((await h.call('models/analysis', { model_id: artifact }, { principal: { kind: 'viewer' } })).error.code, 'tyloinfer/forbidden');
  assert.equal(h.calls.length, 0);
});
test('preserves a core feature-unavailable response instead of inventing empty data', async () => {
  const h = harness(() => Response.json({ error: { code: 'cache_management_unavailable', message: 'Unavailable' } }, { status: 501 }));
  const reply = await h.call('core/read', { resource: 'prefix_cache' });
  assert.equal(reply.ok, false);
  assert.equal(reply.error.details.http_status, 501);
  assert.equal(reply.error.details.core_error.code, 'cache_management_unavailable');
});
