import assert from 'node:assert/strict';
import test from 'node:test';
import { createCoreBridge } from '../../../lib/host/core-bridge.js';

const owner = { principal: { kind: 'owner' } };
const id = '11111111-1111-4111-8111-111111111111';
const artifact = 'a'.repeat(32);
const health = { service: 'tyloinfer-server', protocol_version: '1.0' };
function harness(route, provider = { baseURL: 'http://core.invalid:8090/prefix/v1' }) {
  const calls = [];
  let providerReads = 0;
  const bridge = createCoreBridge({
    getProvider: () => { providerReads++; return provider; },
    resolveCredential: async () => ({ value: 'fixture-core-credential' }),
    fetchImpl: async (url, init) => {
      calls.push({ url, ...init });
      return url.pathname.endsWith('/health') ? Response.json(health) : route(url, init);
    },
  });
  return { calls, providerReads: () => providerReads, call: (method, payload, peer = owner) => bridge(method, payload, undefined, peer) };
}
for (const anthropic_port of [null, 8124]) {
  test('connection metadata uses the configured core, with actual Anthropic availability: ' + anthropic_port, async () => {
    const h = harness(url => { assert.equal(url.pathname, '/prefix/api/v1/runtime/listener'); return Response.json({ port: 8090, anthropic_port }); });
    assert.deepEqual(await h.call('connection/info', {}), { ok: true, value: { base_url: 'http://core.invalid:8090/prefix/v1',
      openai_base_url: 'http://core.invalid:8090/prefix/v1', anthropic_base_url: anthropic_port === null ? null : 'http://core.invalid:8124/prefix/v1',
      anthropic_port, protocol_version: '1.0' } });
    assert.ok(h.calls.every(call => call.method === 'GET'));
  });
}
for (const baseURL of ['http://name:password@core.invalid/v1', 'http://core.invalid/v1?token=fixture', 'http://core.invalid/v1#token']) {
  test('connection metadata never exposes an unsafe configured URL: ' + baseURL, async () => {
    const h = harness(() => assert.fail('network'), { baseURL });
    assert.equal((await h.call('connection/info', {})).error.code, 'tyloinfer/core-incompatible');
    assert.equal(h.calls.length, 0);
  });
}
test('takes explicit Engram companion paths only on model load', async () => {
  const path = '/models/fixture-tables';
  const h = harness((url, init) => {
    assert.equal(url.pathname, '/prefix/api/v1/models/load');
    assert.deepEqual(JSON.parse(init.body), { model: 'fixture', gguf_companions: [path] });
    return Response.json({ operation_id: id, status: 'accepted' }, { status: 202 });
  });
  assert.equal((await h.call('models/load', { request: { model: 'fixture' }, companion_paths: [path] })).ok, true);
  for (const companion_paths of [['relative.gguf'], ['https://fixture.invalid/table'], ['//network/share']]) {
    const reply = await h.call('models/load', { request: { model: 'fixture' }, companion_paths });
    assert.equal(reply.error.code, 'tyloinfer/bad-request');
  }
  assert.equal((await h.call('models/load', { request: { model: 'fixture' }, companion_ids: [artifact], companion_paths: [path] })).error.code, 'tyloinfer/bad-request');
});
test('connection metadata refuses credential values embedded in a path', async () => {
  const h = harness(() => assert.fail('listener'), { baseURL: 'http://core.invalid/fixture-core-credential/v1', apiKeyEnv: 'FIXTURE' });
  const result = await h.call('connection/info', {});
  assert.equal(result.error.code, 'tyloinfer/core-incompatible');
  assert.ok(!JSON.stringify(result).includes('fixture-core-credential'));
});
test('registered GGUF ids are resolved by the core before one fixed load', async () => {
  const h = harness((url, init) => {
    if (init.method === 'GET') {
      assert.equal(url.pathname, '/prefix/api/v1/models/' + artifact + '/companion-reference');
      return Response.json({ model_id: artifact, format: 'gguf', path: '/registered/fixture.gguf' });
    }
    assert.equal(url.pathname, '/prefix/api/v1/models/load');
    assert.deepEqual(JSON.parse(init.body), { model: 'fixture', gguf_companions: ['/registered/fixture.gguf'] });
    return Response.json({ operation_id: id, status: 'accepted' }, { status: 202 });
  });
  assert.equal((await h.call('models/load', { request: { model: 'fixture' }, companion_ids: [artifact] })).value.operation_id, id);
  assert.equal(h.calls.filter(call => call.method === 'POST').length, 1);
});
for (const reference of [{ model_id: artifact, format: 'hf', path: '/registered/fixture' },
  { model_id: 'b'.repeat(32), format: 'gguf', path: '/registered/fixture.gguf' },
  { model_id: artifact, format: 'gguf', path: 'http://outside.invalid/fixture.gguf' }]) {
  test('invalid companion references never reach model loading: ' + JSON.stringify(reference), async () => {
    const h = harness((_url, init) => { assert.equal(init.method, 'GET'); return Response.json(reference); });
    assert.equal((await h.call('models/load', { request: { model: 'fixture' }, companion_ids: [artifact] })).error.code, 'tyloinfer/core-invalid-response');
  });
}
test('an older core without companion lookup is an explicit unsupported feature', async () => {
  const h = harness((_url, init) => { assert.equal(init.method, 'GET'); return Response.json({ detail: 'Not Found' }, { status: 404 }); });
  assert.equal((await h.call('models/load', { request: { model: 'fixture' }, companion_ids: [artifact] })).error.code, 'tyloinfer/operation-not-supported');
});
test('cache directory selection resolves a core directory id and returns the real memory ticket', async () => {
  const h = harness((url, init) => {
    if (init.method === 'GET') {
      assert.equal(url.pathname, '/prefix/api/v1/models/directories');
      assert.equal(url.searchParams.get('directory_id'), artifact);
      return Response.json({ current_id: artifact, current_path: '/registered/cache' });
    }
    assert.equal(url.pathname, '/prefix/api/v1/runtime/memory-policy');
    assert.equal(init.method, 'PUT');
    assert.deepEqual(JSON.parse(init.body), { prefix_directory: '/registered/cache' });
    return Response.json({ operation_id: id, status: 'accepted' }, { status: 202 });
  });
  assert.deepEqual(await h.call('cache/configure-directory', { directory_id: artifact, scope: 'all-loaded-models', confirmed: true }),
    { ok: true, value: { operation_id: id, status: 'accepted' } });
});
for (const kind of ['model.load', 'runtime.context.configure', 'benchmark.inference']) {
  test('terminal job archival preserves native null without touching files: ' + kind, async () => {
    const h = harness((_url, init) => init.method === 'GET' ? Response.json({ id, kind, status: 'succeeded', cancel_requested: false }) : new Response(null, { status: 204 }));
    assert.deepEqual(await h.call('jobs/delete', { job_id: id, confirmed: true }), { ok: true, value: null });
    assert.equal(h.calls.filter(call => call.method === 'DELETE').length, 1);
  });
}
for (const status of ['queued', 'running', 'cancelling']) {
  test('job archival refuses active state without cancelling: ' + status, async () => {
    const h = harness((_url, init) => { assert.equal(init.method, 'GET'); return Response.json({ id, kind: 'model.load', status, cancel_requested: false }); });
    assert.equal((await h.call('jobs/delete', { job_id: id, confirmed: true })).error.code, 'tyloinfer/operation-not-deletable');
  });
}
test('bulk archival uses only the native atomic completed endpoint', async () => {
  const h = harness((url, init) => {
    assert.equal(url.pathname, '/prefix/api/v1/jobs/completed');
    assert.equal(init.method, 'DELETE');
    assert.equal(init.body, undefined);
    return new Response(null, { status: 204 });
  });
  assert.deepEqual(await h.call('jobs/clear-completed', { scope: 'all-completed', confirmed: true }), { ok: true, value: null });
});
for (const [method, payload] of [
  ['connection/info', {}], ['jobs/delete', { job_id: id, confirmed: true }], ['jobs/clear-completed', { scope: 'all-completed', confirmed: true }],
  ['models/load', { request: { model: 'fixture' }, companion_ids: [artifact] }],
  ['cache/configure-directory', { directory_id: artifact, scope: 'all-loaded-models', confirmed: true }],
]) {
  test(method + ' requires owner before provider access', async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call(method, payload, { principal: { kind: 'viewer' } })).error.code, 'tyloinfer/forbidden');
    assert.equal(h.providerReads(), 0);
  });
}
for (const [method, payload] of [
  ['connection/info', { url: 'http://outside.invalid' }], ['jobs/delete', { job_id: id }], ['jobs/delete', { job_id: id, confirmed: true, force: true }],
  ['jobs/clear-completed', { confirmed: true }], ['jobs/clear-completed', { scope: 'all', confirmed: true }],
  ['models/load', { request: { model: 'fixture', gguf_companions: ['/outside'] } }],
  ['models/load', { request: { model: 'fixture' }, companion_ids: ['../outside'] }],
  ['models/load', { request: { model: 'fixture' }, companion_ids: [artifact, artifact] }],
  ['models/load', { request: { model: 'fixture' }, companion_ids: Array.from({ length: 17 }, () => artifact) }],
  ['cache/configure-directory', { directory_id: artifact, confirmed: true }],
  ['cache/configure-directory', { path: '/outside', scope: 'all-loaded-models', confirmed: true }],
]) {
  test('invalid product boundary is refused before network: ' + method + ' ' + JSON.stringify(payload), async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call(method, payload)).error.code, 'tyloinfer/bad-request');
    assert.equal(h.calls.length, 0);
    assert.equal(h.providerReads(), 0);
  });
}
for (const [method, payload] of [['jobs/delete', { job_id: id, confirmed: true }], ['jobs/clear-completed', { scope: 'all-completed', confirmed: true }]]) {
  test(method + ' preserves unknown outcome after DELETE submission', async () => {
    const h = harness((_url, init) => {
      if (init.method === 'GET') return Response.json({ id, kind: 'model.load', status: 'succeeded', cancel_requested: false });
      throw new Error('fixture transport failure');
    });
    const result = await h.call(method, payload);
    assert.equal(result.error.code, 'tyloinfer/outcome-unknown');
    assert.equal(result.error.details.retryable, false);
    assert.equal(h.calls.filter(call => call.method === 'DELETE').length, 1);
  });
}
