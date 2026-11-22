import assert from 'node:assert/strict';
import test from 'node:test';
import { createCoreBridge } from '../../../lib/host/core-bridge.js';

const id = '11111111-1111-4111-8111-111111111111';
const owner = { principal: { kind: 'owner' } };
const health = { service: 'tyloinfer-server', protocol_version: '1.0' };
const json = (value, status = 200) => Response.json(value, { status });
const job = (status = 'running', extra = {}) => ({ id, kind: 'model.load', status, cancel_requested: false, ...extra });
function harness(route, options = {}) {
  const calls = [];
  const bridge = createCoreBridge({
    getProvider: () => ({ baseURL: 'http://core.invalid/v1', ...options.provider }),
    resolveCredential: options.resolveCredential ?? (async () => undefined),
    fetchImpl: async (url, init) => {
      calls.push({ url: String(url), path: url.pathname, ...init });
      return url.pathname.endsWith('/health') ? json(options.health ?? health) : route(url, init);
    },
    ...(options.maxResponseBytes ? { maxResponseBytes: options.maxResponseBytes } : {}),
  });
  return { calls, call: (endpoint, payload, signal = new AbortController().signal, peer = owner) => bridge(endpoint, payload, signal, peer) };
}

test('large expert analysis keeps its full response while other reads retain the shared limit', async () => {
  const artifact = 'a'.repeat(32);
  const expert = { layer: 0, expert: 0, projection: 'gate', parameters: 1638400, stored_bytes: 602112,
    bpw: 2.94, format: 'IQ3_XXS', granularity: 'tensor' };
  const count = 48 * 512 * 3;
  const h = harness(() => json({ experts: Array(count).fill(expert) }));

  const analysis = await h.call('models/analysis', { model_id: artifact });
  assert.equal(analysis.ok, true);
  assert.equal(analysis.value.experts.length, count);
  const ordinary = await h.call('core/read', { resource: 'models' });
  assert.equal(ordinary.error.code, 'tyloinfer/core-invalid-response');
});

for (const endpoint of ['core/read', 'models/load', 'models/unload', 'operations/get', 'operations/cancel']) {
  test(endpoint + ' requires owner before any network request', async () => {
    const h = harness(() => assert.fail('network'));
    const result = await h.call(endpoint, {}, undefined, { principal: { kind: 'viewer' } });
    assert.equal(result.error.code, 'tyloinfer/forbidden');
    assert.equal(h.calls.length, 0);
  });
}

for (const [endpoint, payload] of [
  ['core/read', { resource: '__proto__' }],
  ['core/read', { resource: 'http://untrusted.invalid/' }],
  ['core/read', { resource: 'models', url: 'http://untrusted.invalid/' }],
  ['models/load', { request: {} }],
  ['models/load', { request: { model: ' ' } }],
  ['models/load', { request: { model: 'fixture', context_size: Number.NaN } }],
  ['models/unload', { request: { instance_id: '../../outside' } }],
  ['models/unload', { request: { instance_id: id, force: 'false' } }],
  ['operations/get', { operation_id: '../../outside' }],
  ['operations/cancel', { operation_id: id, force: true }],
  ['arbitrary/proxy', { url: 'http://untrusted.invalid/' }],
]) {
  test('rejects invalid request before network: ' + endpoint + ' ' + JSON.stringify(payload), async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call(endpoint, payload)).error.code, 'tyloinfer/bad-request');
    assert.equal(h.calls.length, 0);
  });
}

test('distinguishes the artifact inventory from loaded runtime models', async () => {
  const h = harness(url => json({ data: [url.pathname] }));
  assert.deepEqual((await h.call('core/read', { resource: 'models' })).value.data, ['/api/v1/models']);
  assert.deepEqual((await h.call('core/read', { resource: 'loaded_models' })).value.data, ['/api/v1/runtime/models']);
  assert.ok(h.calls.every(call => call.method === 'GET' && call.redirect === 'error'));
});

test('preserves native load fields and returns only the accepted operation ticket', async () => {
  const request = { model: 'fixture', context_size: 8192, device_ids: ['cuda:0'],
    kv_memory: { qsa_mode: 'auto', placement: 'auto' }, prefix_cache_enabled: false };
  const h = harness((_url, init) => {
    assert.deepEqual(JSON.parse(init.body), request);
    assert.equal(init.headers.Authorization, 'Bearer example-protected-core-key');
    return json({ operation_id: id, status: 'accepted' }, 202);
  }, { provider: { apiKeyEnv: 'CORE_CREDENTIAL' }, resolveCredential: async () => ({ value: 'example-protected-core-key' }) });
  const result = await h.call('models/load', { request });
  assert.deepEqual(result, { ok: true, value: { operation_id: id, status: 'accepted' } });
  assert.equal(h.calls.filter(call => call.method === 'POST').length, 1);
  assert.equal(h.calls.at(-1).path, '/api/v1/models/load');
  assert.ok(!JSON.stringify(result).includes('example-protected-core-key'));
});

test('unload preserves instance identity and force without widening the request', async () => {
  const h = harness((_url, init) => {
    assert.deepEqual(JSON.parse(init.body), { instance_id: id, force: false });
    return json({ operation_id: id, status: 'accepted' }, 202);
  });
  assert.equal((await h.call('models/unload', { request: { instance_id: id, force: false } })).value.status, 'accepted');
  assert.equal(h.calls.at(-1).path, '/api/v1/models/unload');
});

for (const state of ['queued', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled', 'interrupted']) {
  test('returns native operation state unchanged: ' + state, async () => {
    const h = harness(() => json(job(state)));
    assert.deepEqual((await h.call('operations/get', { operation_id: id })).value, job(state));
    assert.equal(h.calls.filter(call => call.method === 'POST').length, 0);
  });
}

test('cancellation is a request and is not fabricated as terminal completion', async () => {
  const h = harness((_url, init) => json(init.method === 'POST' ? job('cancelling', { cancel_requested: true }) : job()));
  const result = await h.call('operations/cancel', { operation_id: id });
  assert.equal(result.value.status, 'cancelling');
  assert.equal(result.value.cancel_requested, true);
  assert.equal(h.calls.filter(call => call.method === 'POST').length, 1);
  assert.equal(h.calls.at(-1).path, '/api/v1/jobs/' + id + '/cancel');
});

for (const state of ['succeeded', 'failed', 'cancelled', 'interrupted']) {
  test('does not submit cancellation for a terminal job: ' + state, async () => {
    const h = harness(() => json(job(state)));
    assert.equal((await h.call('operations/cancel', { operation_id: id })).error.code, 'tyloinfer/operation-not-cancellable');
    assert.equal(h.calls.filter(call => call.method === 'POST').length, 0);
  });
}

test('does not resend an already requested cancellation', async () => {
  const h = harness(() => json(job('cancelling', { cancel_requested: true })));
  assert.equal((await h.call('operations/cancel', { operation_id: id })).value.status, 'cancelling');
  assert.equal(h.calls.filter(call => call.method === 'POST').length, 0);
});

test('does not expose or cancel unrelated background jobs', async () => {
  const h = harness(() => json(job('running', { kind: 'download.huggingface', payload: { hidden: true } })));
  const result = await h.call('operations/cancel', { operation_id: id });
  assert.equal(result.error.code, 'tyloinfer/operation-not-supported');
  assert.ok(!JSON.stringify(result).includes('hidden'));
  assert.equal(h.calls.filter(call => call.method === 'POST').length, 0);
});

test('write transport failure is an unknown outcome and is never retried', async () => {
  const h = harness(() => { throw new Error('transport failed'); });
  const result = await h.call('models/load', { request: { model: 'fixture' } });
  assert.equal(result.error.code, 'tyloinfer/outcome-unknown');
  assert.equal(result.error.details.retryable, false);
  assert.equal(result.error.details.operation_id, null);
  assert.equal(h.calls.filter(call => call.method === 'POST').length, 1);
});

test('write server failure is also an unknown outcome without retry', async () => {
  const h = harness(() => json({ error: { code: 'internal' } }, 500));
  assert.equal((await h.call('models/load', { request: { model: 'fixture' } })).error.code, 'tyloinfer/outcome-unknown');
  assert.equal(h.calls.filter(call => call.method === 'POST').length, 1);
});

for (const response of [
  () => new Response('not-json', { status: 202 }),
  () => json({ operation_id: 'not-an-operation', status: 'accepted' }, 202),
  () => json({ operation_id: id, status: 'succeeded' }, 202),
  () => json({ operation_id: id, status: 'accepted' }, 200),
]) {
  test('invalid accepted response never invents completion or retries', async () => {
    const h = harness(response);
    assert.equal((await h.call('models/load', { request: { model: 'fixture' } })).error.code, 'tyloinfer/outcome-unknown');
    assert.equal(h.calls.filter(call => call.method === 'POST').length, 1);
  });
}

test('preserves structured core rejection without implying a retry', async () => {
  const h = harness(() => json({ error: { code: 'invalid_context', message: 'Rejected field', details: {} } }, 422));
  const result = await h.call('models/load', { request: { model: 'fixture' } });
  assert.equal(result.error.code, 'tyloinfer/core-rejected');
  assert.equal(result.error.details.http_status, 422);
  assert.equal(result.error.details.core_error.code, 'invalid_context');
  assert.equal(result.error.details.retryable, false);
  assert.equal(h.calls.filter(call => call.method === 'POST').length, 1);
});

test('refuses reflected Host credentials in a core response', async () => {
  const h = harness(() => json({ echo: 'example-protected-core-key' }), {
    provider: { apiKeyEnv: 'CORE_CREDENTIAL' }, resolveCredential: async () => ({ value: 'example-protected-core-key' }),
  });
  const result = await h.call('core/read', { resource: 'models' });
  assert.equal(result.error.code, 'tyloinfer/core-invalid-response');
  assert.ok(!JSON.stringify(result).includes('example-protected-core-key'));
});

test('JSON escaping cannot bypass credential reflection protection', async () => {
  const h = harness(() => new Response('{"echo":"example-\\u0070rotected-core-key"}', { headers: { 'content-type': 'application/json' } }), {
    provider: { apiKeyEnv: 'CORE_CREDENTIAL' }, resolveCredential: async () => ({ value: 'example-protected-core-key' }),
  });
  const result = await h.call('core/read', { resource: 'models' });
  assert.equal(result.error.code, 'tyloinfer/core-invalid-response');
  assert.ok(!JSON.stringify(result).includes('example-protected-core-key'));
});

test('missing credentials fail closed before contacting the core', async () => {
  const h = harness(() => assert.fail('network'), { provider: { apiKeyEnv: 'CORE_CREDENTIAL' } });
  assert.equal((await h.call('models/load', { request: { model: 'fixture' } })).error.code, 'tyloinfer/core-unavailable');
  assert.equal(h.calls.length, 0);
});

test('incompatible core identity prevents every write', async () => {
  const h = harness(() => assert.fail('write'), { health: { service: 'another-service', protocol_version: '1.0' } });
  assert.equal((await h.call('models/load', { request: { model: 'fixture' } })).error.code, 'tyloinfer/core-incompatible');
  assert.equal(h.calls.length, 1);
});

test('an already aborted caller never submits a model operation', async () => {
  const h = harness(() => assert.fail('network'));
  const controller = new AbortController(); controller.abort();
  assert.equal((await h.call('models/load', { request: { model: 'fixture' } }, controller.signal)).error.code, 'tyloinfer/cancelled');
  assert.equal(h.calls.length, 0);
});

test('caller cancellation also bounds pending credential resolution', async () => {
  const h = harness(() => assert.fail('network'), { provider: { apiKeyEnv: 'CORE_CREDENTIAL' }, resolveCredential: () => new Promise(() => {}) });
  const controller = new AbortController();
  const pending = h.call('models/load', { request: { model: 'fixture' } }, controller.signal);
  controller.abort();
  assert.equal((await pending).error.code, 'tyloinfer/cancelled');
  assert.equal(h.calls.length, 0);
});

test('caller abort after POST submission does not claim server cancellation', async () => {
  const controller = new AbortController();
  const h = harness(() => { controller.abort(); throw controller.signal.reason; });
  const result = await h.call('models/load', { request: { model: 'fixture' } }, controller.signal);
  assert.equal(result.error.code, 'tyloinfer/outcome-unknown');
  assert.equal(h.calls.filter(call => call.method === 'POST').length, 1);
});

test('bounded response parsing rejects oversized core data', async () => {
  const h = harness(() => json({ data: ['x'.repeat(256)] }), { maxResponseBytes: 128 });
  assert.equal((await h.call('core/read', { resource: 'models' })).error.code, 'tyloinfer/core-invalid-response');
});
