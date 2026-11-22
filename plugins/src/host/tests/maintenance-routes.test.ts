import assert from 'node:assert/strict';
import test from 'node:test';
import { createCoreBridge } from '../../../lib/host/core-bridge.js';

const id = '11111111-1111-4111-8111-111111111111';
const otherId = '22222222-2222-4222-8222-222222222222';
const namespace = 'a'.repeat(64);
const block = 'b'.repeat(64);
const directory = 'c'.repeat(32);
const owner = { principal: { kind: 'owner' } };
const json = (value, status = 200) => Response.json(value, { status });
const stamp = '2026-01-01T00:00:00Z';
const nativeJob = { id, kind: 'benchmark.inference', status: 'succeeded', payload: {}, progress: 1, progress_data: {}, cancel_requested: false,
  result: { fixture: true }, error: null, created_at: stamp, updated_at: stamp, started_at: stamp, completed_at: stamp };
const event = (sequence, extra = {}) => ({ job_id: id, sequence, type: 'log', level: 'info', message: 'fixture', data: {}, created_at: stamp, ...extra });
const directoryStatus = { directory: null, actual_directory: 'fixture-downloads', default_directory: 'fixture-downloads', evict_prefix_cache: false,
  disk_total_bytes: null, disk_free_bytes: null };
const validCalls = [
  ['cache/clear', { instance_id: id, confirmed: true }, 'POST', '/api/v1/runtime/cache/clear', { instance_id: id }, { status: 'ok', released_snapshots: 2, image_decode_cache_released_bytes: 3 }],
  ['cache/trim', { instance_id: id, target_bytes: 0, confirmed: true }, 'POST', '/api/v1/runtime/cache/trim', { instance_id: id, target_bytes: 0 }, { status: 'ok', released_bytes: 2, target_bytes: 0 }],
  ['cache/purge', { scope: 'namespace', namespace, confirmed: true }, 'POST', '/api/v1/runtime/cache/purge', { namespace }, { status: 'ok', released_bytes: 2, removed_blocks: 1, failed_blocks: 0 }],
  ['cache/purge', { scope: 'all', confirmed: true }, 'POST', '/api/v1/runtime/cache/purge', { namespace: null }, { status: 'ok', released_bytes: 2, removed_blocks: 1, failed_blocks: 1 }],
  ['downloads/configure-directory', { request: { directory_id: directory }, confirmed: true }, 'PUT', '/api/v1/hub/download-directory', { directory_id: directory }, directoryStatus],
  ['downloads/configure-directory', { request: { directory_id: null }, confirmed: true }, 'PUT', '/api/v1/hub/download-directory', { directory_id: null }, directoryStatus],
  ['downloads/configure-directory', { request: { evict_prefix_cache: false }, confirmed: true }, 'PUT', '/api/v1/hub/download-directory', { evict_prefix_cache: false }, directoryStatus],
];
function harness(route, options = {}) {
  const calls = [];
  let providerReads = 0;
  const bridge = createCoreBridge({ getProvider: () => { providerReads++; return { baseURL: 'http://core.invalid/v1' }; }, resolveCredential: async () => undefined,
    fetchImpl: async (url, init) => { calls.push({ url, ...init }); return url.pathname === '/health' ? json({ service: 'tyloinfer-server', protocol_version: '1.0' }) : route(url, init); }, ...options });
  return { calls, get providerReads() { return providerReads; }, call: (endpoint, payload, peer = owner, signal) => bridge(endpoint, payload, signal, peer) };
}
for (const [endpoint, payload, method, path, body, response] of validCalls) {
  test('confirmed maintenance preserves the exact native call: ' + endpoint + ' ' + JSON.stringify(body), async () => {
    const h = harness((url, init) => { assert.equal(url.pathname, path); assert.equal(init.method, method); assert.deepEqual(JSON.parse(init.body), body); return json(response); });
    assert.deepEqual(await h.call(endpoint, payload), { ok: true, value: response });
    assert.equal(h.calls.length, 2);
    assert.equal(h.calls.filter(call => call.method !== 'GET').length, 1);
    assert.ok(!h.calls.at(-1).body.includes('confirmed'));
    assert.ok(!h.calls.at(-1).body.includes('scope'));
  });
  for (const confirmed of [undefined, false, 'true', null]) {
    test('rejects absent or non-boolean confirmation before provider access: ' + endpoint + ' ' + JSON.stringify(body) + ' ' + confirmed, async () => {
      const h = harness(() => assert.fail('network'));
      const invalid = { ...payload, ...(confirmed === undefined ? {} : { confirmed }) };
      if (confirmed === undefined) delete invalid.confirmed;
      assert.equal((await h.call(endpoint, invalid)).error.code, 'tyloinfer/bad-request');
      assert.equal(h.providerReads, 0);
      assert.equal(h.calls.length, 0);
    });
  }
}
for (const endpoint of ['cache/clear', 'cache/trim', 'cache/purge', 'cache/text', 'downloads/configure-directory', 'jobs/get', 'jobs/events']) {
  test('maintenance requires owner before any provider or network: ' + endpoint, async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call(endpoint, {}, { principal: { kind: 'viewer' } })).error.code, 'tyloinfer/forbidden');
    assert.equal(h.providerReads, 0);
  });
}
for (const [endpoint, payload] of [
  ['cache/clear', { confirmed: true }], ['cache/clear', { instance_id: null, confirmed: true }],
  ['cache/clear', { instance_id: id, scope: 'all', confirmed: true }], ['cache/clear', { instance_id: id, force: true, confirmed: true }],
  ['cache/trim', { instance_id: id, confirmed: true }], ['cache/trim', { instance_id: id, target_bytes: -1, confirmed: true }],
  ['cache/trim', { instance_id: id, target_bytes: Number.MAX_SAFE_INTEGER + 1, confirmed: true }], ['cache/trim', { instance_id: id, target_bytes: '0', confirmed: true }],
  ['cache/purge', { confirmed: true }], ['cache/purge', { scope: 'namespace', confirmed: true }],
  ['cache/purge', { scope: 'namespace', namespace: null, confirmed: true }], ['cache/purge', { scope: 'all', namespace: null, confirmed: true }],
  ['cache/purge', { scope: 'all', namespace, confirmed: true }], ['cache/purge', { scope: 'namespace', namespace: '../outside', confirmed: true }],
  ['cache/text', { namespace, block: '../outside' }], ['cache/text', { namespace: namespace.toUpperCase(), block }],
  ['cache/text', { namespace, block, limit: 8193 }], ['cache/text', { namespace, block, limit: 0 }],
  ['cache/text', { namespace, block, offset: -1 }], ['cache/text', { namespace, block, url: '/outside' }],
  ['downloads/configure-directory', { request: {}, confirmed: true }], ['downloads/configure-directory', { request: { path: '/outside' }, confirmed: true }],
  ['downloads/configure-directory', { request: { download_directory: '/outside' }, confirmed: true }],
  ['downloads/configure-directory', { request: { directory_id: '../outside' }, confirmed: true }],
  ['downloads/configure-directory', { request: { evict_prefix_cache: 'true' }, confirmed: true }],
  ['jobs/get', { job_id: '../outside' }], ['jobs/get', { operation_id: id }], ['jobs/get', { job_id: id, method: 'DELETE' }],
  ['jobs/events', { job_id: id, limit: 1001 }], ['jobs/events', { job_id: id, limit: 0 }], ['jobs/events', { job_id: id, after: 0.5 }],
  ['jobs/events', { job_id: id, tail: 'true' }], ['jobs/events', { job_id: id, path: '/outside' }],
  ['jobs/delete', { job_id: id, confirmed: true, force: true }], ['jobs/retry', { job_id: id, confirmed: true }], ['jobs/cancel', { job_id: id, confirmed: true }],
]) {
  test('rejects invalid maintenance scope or field before network: ' + endpoint + ' ' + JSON.stringify(payload), async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call(endpoint, payload)).error.code, 'tyloinfer/bad-request');
    assert.equal(h.providerReads, 0);
    assert.equal(h.calls.length, 0);
  });
}
for (const [payload, response, params] of [
  [{ namespace, block }, { available: false, reason: 'text_not_saved' }, {}],
  [{ namespace, block, offset: 8, limit: 4 }, { available: true, text: 'fixture text', total_tokens: 20, offset: 8, next_offset: 12 }, { offset: '8', limit: '4' }],
]) {
  test('cache text preserves native availability and token offsets without recovery actions', async () => {
    const h = harness((url, init) => { assert.equal(url.pathname, '/api/v1/runtime/cache/entries/' + namespace + '/' + block + '/text'); assert.deepEqual(Object.fromEntries(url.searchParams), params); assert.equal(init.method, 'GET'); return json(response); });
    assert.deepEqual(await h.call('cache/text', payload), { ok: true, value: response });
    assert.equal(h.calls.filter(call => call.method !== 'GET').length, 0);
  });
}
test('job details can read a log-visible job but cannot grant write authority', async () => {
  const h = harness((url, init) => { assert.equal(url.pathname, '/api/v1/jobs/' + id); assert.equal(init.method, 'GET'); return json(nativeJob); });
  assert.deepEqual(await h.call('jobs/get', { job_id: id }), { ok: true, value: nativeJob });
  assert.equal((await h.call('operations/cancel', { operation_id: id })).error.code, 'tyloinfer/operation-not-supported');
  assert.equal(h.calls.filter(call => call.method !== 'GET').length, 0);
});
test('job events preserve raw identities, types, cursor order and explicit tail selection', async () => {
  const response = { data: [event(4), event(5, { type: 'artifact', data: { fixture: true } })] };
  const h = harness((url, init) => { assert.equal(url.pathname, '/api/v1/jobs/' + id + '/events'); assert.deepEqual(Object.fromEntries(url.searchParams), { after: '3', limit: '2', tail: 'true' }); assert.equal(init.method, 'GET'); return json(response); });
  assert.deepEqual(await h.call('jobs/events', { job_id: id, after: 3, limit: 2, tail: true }), { ok: true, value: response });
});
for (const [endpoint, payload, response] of [
  ['cache/text', { namespace, block }, { available: true, text: 1 }],
  ['cache/text', { namespace, block }, { available: false }],
  ['jobs/get', { job_id: id }, { ...nativeJob, id: otherId }],
  ['jobs/get', { job_id: id }, { ...nativeJob, progress: 2 }],
  ['jobs/events', { job_id: id }, { data: [event(2), event(1)] }],
  ['jobs/events', { job_id: id }, { data: [event(1, { job_id: otherId })] }],
  ['jobs/events', { job_id: id, after: 2 }, { data: [event(1)] }],
]) {
  test('rejects malformed read data rather than inventing empty state: ' + endpoint + ' ' + JSON.stringify(response), async () => {
    const h = harness(() => json(response));
    assert.equal((await h.call(endpoint, payload)).error.code, 'tyloinfer/core-invalid-response');
  });
}
for (const [endpoint, payload, , , , response] of validCalls) {
  test('unconfirmed write outcome is never retried: ' + endpoint + ' ' + JSON.stringify(payload), async () => {
    const h = harness(() => { throw new Error('transport'); });
    const reply = await h.call(endpoint, payload);
    assert.equal(reply.error.code, 'tyloinfer/outcome-unknown');
    assert.equal(reply.error.details.retryable, false);
    assert.equal(reply.error.details.operation_id, null);
    assert.equal(h.calls.filter(call => call.method !== 'GET').length, 1);
  });
  test('malformed maintenance success does not fabricate completion: ' + endpoint + ' ' + JSON.stringify(payload), async () => {
    const h = harness(() => json({ ...response, status: 'accepted', actual_directory: null, removed_blocks: 'invalid', target_bytes: -1 }));
    assert.equal((await h.call(endpoint, payload)).error.code, 'tyloinfer/outcome-unknown');
    assert.equal(h.calls.filter(call => call.method !== 'GET').length, 1);
  });
}
test('busy purge retains the native conflict and does not force or retry', async () => {
  const h = harness(() => json({ error: { code: 'runtime_busy', message: 'Wait for current work', retryable: true } }, 409));
  const reply = await h.call('cache/purge', { scope: 'all', confirmed: true });
  assert.equal(reply.error.code, 'tyloinfer/core-rejected');
  assert.equal(reply.error.details.http_status, 409);
  assert.equal(reply.error.details.core_error.code, 'runtime_busy');
  assert.equal(reply.error.details.retryable, false);
  assert.equal(h.calls.filter(call => call.method !== 'GET').length, 1);
});
const delayed = (response, delay) => (_url, init) => new Promise((resolve, reject) => {
  const timer = setTimeout(() => resolve(json(response)), delay);
  init.signal.addEventListener('abort', () => { clearTimeout(timer); reject(init.signal.reason); }, { once: true });
});
test('maintenance has its own configurable deadline instead of the ordinary read deadline', async () => {
  const h = harness(delayed({ status: 'ok', released_bytes: 0, removed_blocks: 0, failed_blocks: 0 }, 20), { timeoutMs: 1, maintenanceTimeoutMs: 500 });
  assert.equal((await h.call('cache/purge', { scope: 'all', confirmed: true })).ok, true);
});
test('a maintenance timeout after POST is unknown and not an invented operation', async () => {
  const h = harness(delayed({ status: 'ok', released_bytes: 0, removed_blocks: 0, failed_blocks: 0 }, 1000), { maintenanceTimeoutMs: 20 });
  const reply = await h.call('cache/purge', { scope: 'all', confirmed: true });
  assert.equal(reply.error.code, 'tyloinfer/outcome-unknown');
  assert.equal(h.calls.filter(call => call.method !== 'GET').length, 1);
});
test('caller cancellation after maintenance submission does not claim cancellation of the action', async () => {
  const controller = new AbortController();
  const h = harness(() => { controller.abort(); throw controller.signal.reason; });
  assert.equal((await h.call('cache/clear', { instance_id: id, confirmed: true }, owner, controller.signal)).error.code, 'tyloinfer/outcome-unknown');
});
