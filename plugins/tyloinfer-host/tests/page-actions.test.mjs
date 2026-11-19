import assert from 'node:assert/strict';
import test from 'node:test';
import { createCoreBridge } from '../core-bridge.js';

const id = '11111111-1111-4111-8111-111111111111';
const directory = '1234567890abcdef1234567890abcdef';
const owner = { principal: { kind: 'owner' } };
const job = (kind, status = 'running', extra = {}) => ({ id, kind, status, cancel_requested: false, ...extra });
const json = (value, status = 200) => Response.json(value, { status });
function harness(route, kinds = ['download.huggingface', 'download.modelscope']) {
  const calls = [];
  const bridge = createCoreBridge({ getProvider: () => ({ baseURL: 'http://core.invalid/v1' }), resolveCredential: async () => undefined,
    fetchImpl: async (url, init) => {
      calls.push({ url, ...init });
      if (url.pathname === '/health') return json({ service: 'tyloinfer-server', protocol_version: '1.0' });
      if (url.pathname === '/api/v1/jobs/kinds') return json({ data: kinds.map(kind => ({ kind })) });
      return route(url, init);
    } });
  return { calls, call: (method, payload, peer = owner) => bridge(method, payload, undefined, peer) };
}
for (const endpoint of ['hub/search', 'hub/info', 'hub/official', 'models/register', 'downloads/create', 'downloads/get', 'downloads/cancel', 'service/configure', 'service/get', 'service/cancel']) {
  test(endpoint + ' refuses non-owner before any network', async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call(endpoint, {}, { principal: { kind: 'viewer' } })).error.code, 'tyloinfer/forbidden');
    assert.equal(h.calls.length, 0);
  });
}
for (const [endpoint, payload, path, params] of [
  ['hub/search', { provider: 'huggingface', query: 'model & version', formats: ['tq', 'gguf'], sort: 'downloads', direction: 'desc', cursor: 'a'.repeat(16), limit: 40 }, '/api/v1/hub/models',
    { provider: 'huggingface', query: 'model & version', formats: 'tq,gguf', sort: 'downloads', direction: 'desc', cursor: 'a'.repeat(16), limit: '40', browse: 'true' }],
  ['hub/search', { provider: 'modelscope', query: '', formats: [] }, '/api/v1/hub/models', { provider: 'modelscope', query: '', browse: 'true' }],
  ['hub/info', { provider: 'modelscope', repo_id: 'example/model', revision: 'branch/a&b' }, '/api/v1/hub/models/modelscope/example/model', { revision: 'branch/a&b' }],
  ['hub/official', { refresh: true }, '/api/v1/hub/official', { refresh: 'true' }],
]) {
  test(endpoint + ' constructs a fixed metadata GET and encoded named fields', async () => {
    const h = harness((url, init) => { assert.equal(url.pathname, path); assert.equal(init.method, 'GET'); assert.deepEqual(Object.fromEntries(url.searchParams), params); return json({ data: [] }); });
    assert.equal((await h.call(endpoint, payload)).ok, true);
    assert.equal(h.calls.filter(call => call.method !== 'GET').length, 0);
  });
}
test('registers only the core-issued directory id without opening or loading anything', async () => {
  const h = harness((url, init) => { assert.equal(url.pathname, '/api/v1/models/directories/register'); assert.equal(init.method, 'POST'); assert.deepEqual(JSON.parse(init.body), { directory_id: directory }); return json({ data: [] }); });
  assert.deepEqual(await h.call('models/register', { directory_id: directory }), { ok: true, value: { data: [] } });
  assert.equal(h.calls.filter(call => call.method === 'POST').length, 1);
});
for (const provider of ['huggingface', 'modelscope']) {
  test('preserves the native ' + provider + ' download JobResource without inventing completion', async () => {
    const request = { repo_id: 'example/model', destination: 'models/example/model/variant', revision: 'main', include: ['*.gguf'], expected_bytes: null, max_workers: 2 };
    const native = job('download.' + provider, 'queued');
    const h = harness((url, init) => { assert.equal(url.pathname, '/api/v1/jobs'); assert.equal(init.method, 'POST'); assert.deepEqual(JSON.parse(init.body), { kind: 'download.' + provider, payload: request }); return json(native, 202); });
    assert.deepEqual(await h.call('downloads/create', { provider, request }), { ok: true, value: native });
    assert.equal(h.calls.filter(call => call.method === 'POST').length, 1);
  });
  test('tracks and requests cancellation only for ' + provider + ' downloads', async () => {
    const h = harness((_url, init) => json(job('download.' + provider, init.method === 'POST' ? 'cancelling' : 'running', { cancel_requested: init.method === 'POST' })));
    assert.equal((await h.call('downloads/get', { operation_id: id })).value.kind, 'download.' + provider);
    const reply = await h.call('downloads/cancel', { operation_id: id });
    assert.equal(reply.value.status, 'cancelling');
    assert.equal(reply.value.cancel_requested, true);
    assert.equal(h.calls.filter(call => call.method === 'POST').length, 1);
  });
}
test('a missing download handler fails before submitting or creating download directories', async () => {
  const h = harness(() => assert.fail('POST'), []);
  const reply = await h.call('downloads/create', { provider: 'huggingface', request: { repo_id: 'example/model', destination: 'model' } });
  assert.equal(reply.error.code, 'tyloinfer/operation-not-supported');
  assert.equal(h.calls.filter(call => call.method === 'POST').length, 0);
});
for (const [resource, request, response] of [
  ['inference_policy', { mtp_enabled: false }, { mtp_enabled: false }],
  ['model_aliases', { aliases: { model: 'alias' } }, { aliases: { model: 'alias' } }],
  ['context_policy', { max_context_size: null }, { max_context_size: null, fallback_context_size: 32768 }],
]) {
  test('saves only the native synchronous ' + resource + ' value through PUT', async () => {
    const h = harness((url, init) => { assert.equal(url.pathname, '/api/v1/runtime/' + resource.replaceAll('_', '-')); assert.equal(init.method, 'PUT'); assert.deepEqual(JSON.parse(init.body), request); return json(response); });
    assert.deepEqual(await h.call('service/configure', { resource, request }), { ok: true, value: response });
    assert.equal(h.calls.filter(call => call.method === 'PUT').length, 1);
  });
}
test('memory policy returns a native accepted operation and not synchronous completion', async () => {
  const h = harness((_url, init) => { assert.equal(init.method, 'PUT'); assert.deepEqual(JSON.parse(init.body), { prefix_ram_limit_bytes: 0, model_limit_bytes: null }); return json({ operation_id: id, status: 'accepted' }, 202); });
  assert.equal((await h.call('service/configure', { resource: 'memory_policy', request: { prefix_ram_limit_bytes: 0, model_limit_bytes: null } })).value.status, 'accepted');
});
test('service operation tracking is restricted to native memory configuration', async () => {
  const h = harness(() => json(job('runtime.memory.configure')));
  assert.equal((await h.call('service/get', { operation_id: id })).value.kind, 'runtime.memory.configure');
});
for (const endpoint of ['downloads/get', 'downloads/cancel', 'service/get', 'service/cancel']) {
  test(endpoint + ' cannot read or cancel unrelated jobs', async () => {
    const h = harness(() => json(job('benchmark.inference', 'running', { payload: { hidden: true } })));
    const reply = await h.call(endpoint, { operation_id: id });
    assert.equal(reply.error.code, 'tyloinfer/operation-not-supported');
    assert.ok(!JSON.stringify(reply).includes('hidden'));
    assert.equal(h.calls.filter(call => call.method === 'POST').length, 0);
  });
}
for (const [endpoint, payload] of [
  ['hub/search', { provider: 'custom', query: 'model' }], ['hub/search', { provider: 'huggingface', query: 'x', url: 'http://untrusted.invalid' }],
  ['hub/search', { provider: 'huggingface', query: 'x', formats: ['hf'] }], ['hub/search', { provider: 'huggingface', query: 'x', limit: 101 }],
  ['hub/search', { provider: 'huggingface', query: 'x', cursor: '../cursor' }], ['hub/search', { provider: 'huggingface', query: 'x', direction: 'invalid' }],
  ['hub/info', { provider: 'huggingface', repo_id: '../model' }], ['hub/info', { provider: 'huggingface', repo_id: 'example/..' }],
  ['hub/info', { provider: 'huggingface', repo_id: 'http://untrusted.invalid' }], ['hub/official', { refresh: 'true' }],
  ['models/register', { path: '/outside' }], ['models/register', { directory_id: directory, path: '/outside' }],
  ['downloads/create', { provider: 'huggingface', kind: 'benchmark.inference', request: { repo_id: 'example/model', destination: 'model' } }],
  ['downloads/create', { provider: 'huggingface', request: { repo_id: 'example/model', destination: '../outside' } }],
  ['downloads/create', { provider: 'huggingface', request: { repo_id: 'example/model', destination: '/outside' } }],
  ['downloads/create', { provider: 'huggingface', request: { repo_id: 'example/model', destination: 'X:\\outside' } }],
  ['downloads/create', { provider: 'huggingface', request: { repo_id: 'example/model', destination: 'model', api_key: 'not-accepted' } }],
  ['downloads/create', { provider: 'modelscope', request: { repo_id: 'example/model', destination: 'model', max_workers: 17 } }],
  ['downloads/create', { provider: 'modelscope', request: { repo_id: 'example/model', destination: 'model', include: [1] } }],
  ['downloads/create', { provider: 'modelscope', request: { repo_id: 'example/model', destination: 'model', expected_bytes: -1 } }],
  ['downloads/cancel', { operation_id: '../jobs' }], ['service/cancel', { operation_id: id, force: true }],
  ['service/configure', { resource: 'authentication_policy', request: { enabled: false } }],
  ['service/configure', { resource: 'listener', request: { port: 1 } }],
  ['service/configure', { resource: 'inference_policy', request: { mtp_enabled: 'false' } }],
  ['service/configure', { resource: 'model_aliases', request: { aliases: { model: 1 } } }],
  ['service/configure', { resource: 'context_policy', request: { max_context_size: 511 } }],
  ['service/configure', { resource: 'context_policy', request: { max_context_size: null, apply_global_cap: true } }],
  ['service/configure', { resource: 'memory_policy', request: { prefix_directory: '/outside' } }],
  ['service/configure', { resource: 'memory_policy', request: { model_limit_bytes: 0 } }],
  ['service/configure', { resource: 'memory_policy', request: {} }], ['jobs/create', { kind: 'benchmark.inference' }],
]) {
  test('rejects unexposed page action before network: ' + endpoint + ' ' + JSON.stringify(payload), async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call(endpoint, payload)).error.code, 'tyloinfer/bad-request');
    assert.equal(h.calls.length, 0);
  });
}
for (const [endpoint, payload] of [
  ['models/register', { directory_id: directory }],
  ['downloads/create', { provider: 'huggingface', request: { repo_id: 'example/model', destination: 'model' } }],
  ['service/configure', { resource: 'inference_policy', request: { mtp_enabled: false } }],
  ['service/configure', { resource: 'memory_policy', request: { model_limit_bytes: null } }],
]) {
  test(endpoint + ' treats connection loss after write as unknown and never retries', async () => {
    const h = harness(() => { throw new Error('transport'); });
    const reply = await h.call(endpoint, payload);
    assert.equal(reply.error.code, 'tyloinfer/outcome-unknown');
    assert.equal(reply.error.details.retryable, false);
    assert.equal(h.calls.filter(call => ['POST', 'PUT'].includes(call.method)).length, 1);
  });
}
test('a wrong-kind download acceptance is unknown, not an invented ticket', async () => {
  const h = harness(() => json(job('benchmark.inference'), 202));
  assert.equal((await h.call('downloads/create', { provider: 'huggingface', request: { repo_id: 'example/model', destination: 'model' } })).error.code, 'tyloinfer/outcome-unknown');
  assert.equal(h.calls.filter(call => call.method === 'POST').length, 1);
});
test('synchronous policy responses must contain the actual policy fields', async () => {
  const h = harness(() => json({ status: 'accepted' }));
  assert.equal((await h.call('service/configure', { resource: 'inference_policy', request: { mtp_enabled: true } })).error.code, 'tyloinfer/outcome-unknown');
});
test('the page action body transport limit is checked before network', async () => {
  const h = harness(() => assert.fail('network'));
  const request = { repo_id: 'example/model', destination: 'model', include: Array.from({ length: 100 }, () => 'x'.repeat(1024)) };
  assert.equal((await h.call('downloads/create', { provider: 'huggingface', request })).error.code, 'tyloinfer/bad-request');
  assert.equal(h.calls.length, 0);
});
