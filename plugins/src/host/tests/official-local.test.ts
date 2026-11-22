import assert from 'node:assert/strict';
import test from 'node:test';
import { createCoreBridge } from '../../../lib/host/core-bridge.js';

const owner = { principal: { kind: 'owner' } };
const system = { platform: 'fixture', machine: 'fixture', backend: 'cpu' };
const snapshot = refreshing => ({ system, data: [{ id: 'fixture', name: 'Fixture', sources: [], variants: [] }], refreshing });
function harness(route) {
  const calls = [];
  let providers = 0;
  const bridge = createCoreBridge({
    getProvider: () => { providers++; return { baseURL: 'http://core.invalid/prefix/v1' }; },
    resolveCredential: async () => undefined,
    fetchImpl: async (url, init) => {
      calls.push({ url, ...init });
      assert.equal(init.method, 'GET');
      assert.equal(init.body, undefined);
      if (url.pathname === '/prefix/health') return Response.json({ service: 'tyloinfer-server', protocol_version: '1.0' });
      return route(url, init);
    },
  });
  return { calls, providers: () => providers, call: (payload = {}, peer = owner) => bridge('hub/official-local', payload, undefined, peer) };
}

for (const refreshing of [false, true]) {
  test('local catalog uses one fixed snapshot GET and preserves existing refresh state: ' + refreshing, async () => {
    const value = snapshot(refreshing);
    const h = harness(url => {
      assert.equal(url.pathname, '/prefix/api/v1/hub/official/local');
      assert.equal(url.search, '');
      return Response.json(value);
    });
    assert.deepEqual(await h.call(), { ok: true, value });
    assert.deepEqual(h.calls.map(call => call.url.pathname), ['/prefix/health', '/prefix/api/v1/hub/official/local']);
  });
}

test('local catalog requires owner before provider access', async () => {
  const h = harness(() => assert.fail('network'));
  assert.equal((await h.call({}, { principal: { kind: 'viewer' } })).error.code, 'tyloinfer/forbidden');
  assert.equal(h.providers(), 0);
  assert.equal(h.calls.length, 0);
});

for (const payload of [{ refresh: false }, { local_only: true }, { provider: 'huggingface' }, { url: 'http://outside.invalid' }, { path: '/outside' }, null, []]) {
  test('local catalog refuses every non-empty or non-object payload before network: ' + JSON.stringify(payload), async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call(payload)).error.code, 'tyloinfer/bad-request');
    assert.equal(h.providers(), 0);
    assert.equal(h.calls.length, 0);
  });
}

for (const value of [{ detail: 'Not Found' }, { error: { code: 'route_not_found', message: 'Missing route' } }]) {
  test('an older core is unsupported without falling back to the network catalog: ' + JSON.stringify(value), async () => {
    const h = harness(url => {
      assert.equal(url.pathname, '/prefix/api/v1/hub/official/local');
      return Response.json(value, { status: 404 });
    });
    const result = await h.call();
    assert.equal(result.error.code, 'tyloinfer/operation-not-supported');
    assert.equal(result.error.details.retryable, false);
    assert.equal(h.calls.length, 2);
    assert.ok(h.calls.every(call => !call.url.pathname.endsWith('/official')));
  });
}

test('an unavailable local catalog provider preserves the native rejection without a fallback', async () => {
  const h = harness(() => Response.json({ error: { code: 'local_official_catalog_unavailable', message: 'Unavailable' } }, { status: 501 }));
  const result = await h.call();
  assert.equal(result.error.code, 'tyloinfer/core-rejected');
  assert.equal(result.error.details.http_status, 501);
  assert.equal(result.error.details.core_error.code, 'local_official_catalog_unavailable');
  assert.equal(h.calls.length, 2);
});

for (const value of [{}, { system, data: [], refreshing: 'false' }, { system: {}, data: [], refreshing: false },
  { system, data: [null], refreshing: false }, { system, data: [{}], refreshing: false }]) {
  test('invalid local catalog data is not accepted as an empty snapshot: ' + JSON.stringify(value), async () => {
    const h = harness(() => Response.json(value));
    assert.equal((await h.call()).error.code, 'tyloinfer/core-invalid-response');
    assert.equal(h.calls.length, 2);
  });
}

test('local snapshot transport errors remain read failures and are never retried', async () => {
  const h = harness(() => { throw new Error('fixture transport failure'); });
  const result = await h.call();
  assert.equal(result.error.code, 'tyloinfer/core-unavailable');
  assert.equal(h.calls.length, 2);
});
