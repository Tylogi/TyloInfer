import assert from 'node:assert/strict';
import test from 'node:test';
import { createCoreBridge } from '../../../lib/host/core-bridge.js';

const artifact = '1234567890abcdef1234567890abcdef';
const owner = { principal: { kind: 'owner' } };
const health = { service: 'tyloinfer-server', protocol_version: '1.0' };
const endpoints = ['models/cache-profile', 'models/directory'];
const profile = { max_context: 16384, fixed_bytes: 128, fixed_components: [], components: [
  { group: 'KV', bytes_per_row: 64, tokens_per_row: 1, allocation: 'exact', minimum_rows: 0, row_rounding: 'ceil', active_after: 0 },
] };
const directory = { current_id: artifact, current_name: 'fixture', current_path: '/fixture/models', parent_id: null,
  model_file_count: 1, files: [{ name: 'fixture.tq', byte_size: 128 }], can_open_directory: false, can_open_in_finder: false, data: [] };
function harness(route, options = {}) {
  const calls = [];
  let providerReads = 0;
  let credentialReads = 0;
  const bridge = createCoreBridge({
    getProvider: () => { providerReads++; return { baseURL: 'http://core.invalid/prefix/v1', ...(options.provider ?? {}) }; },
    resolveCredential: async () => { credentialReads++; return options.credential; },
    fetchImpl: async (url, init) => {
      calls.push({ url, ...init });
      return url.pathname.endsWith('/health') ? Response.json(options.health ?? health) : route(url, init);
    },
    ...(options.maxResponseBytes ? { maxResponseBytes: options.maxResponseBytes } : {}),
  });
  return { calls, counts: () => ({ providerReads, credentialReads }),
    call: (endpoint, payload, peer = owner, signal) => bridge(endpoint, payload, signal, peer) };
}
for (const endpoint of endpoints) {
  test(endpoint + ' uses only the fixed GET with the configured base prefix', async () => {
    const h = harness(url => Response.json({ path: url.pathname, query: url.search }));
    const result = await h.call(endpoint, { model_id: artifact });
    assert.deepEqual(result, { ok: true, value: { path: '/prefix/api/v1/models/' + artifact + '/' + endpoint.split('/')[1], query: '' } });
    assert.equal(h.calls.length, 2);
    assert.ok(h.calls.every(call => call.method === 'GET' && call.redirect === 'error' && call.body === undefined));
  });
  for (const peer of [{ principal: { kind: 'viewer' } }, {}, { principal: { kind: 'agent' } }]) {
    test(endpoint + ' rejects a non-owner before provider and credentials: ' + JSON.stringify(peer), async () => {
      const h = harness(() => assert.fail('network'));
      assert.equal((await h.call(endpoint, { model_id: artifact }, peer)).error.code, 'tyloinfer/forbidden');
      assert.deepEqual(h.counts(), { providerReads: 0, credentialReads: 0 });
      assert.equal(h.calls.length, 0);
    });
  }
  for (const payload of [null, [], {}, { model_id: null }, { model_id: 7 }, { model_id: 'a'.repeat(31) },
    { model_id: 'a'.repeat(33) }, { model_id: artifact.toUpperCase() }, { model_id: '11111111-1111-4111-8111-111111111111' },
    { model_id: 'model-name' }, { model_id: '../outside' }, { model_id: '%2e%2e%2foutside' }, { model_id: artifact + '?query=true' },
    { model_id: artifact + '\u0000' }, ...['path', 'url', 'query', 'method', 'directory_id'].map(key => ({ model_id: artifact, [key]: '/outside' }))]) {
    test(endpoint + ' rejects unsupported identity or fields before provider: ' + JSON.stringify(payload), async () => {
      const h = harness(() => assert.fail('network'));
      assert.equal((await h.call(endpoint, payload)).error.code, 'tyloinfer/bad-request');
      assert.equal(h.calls.length, 0);
      assert.deepEqual(h.counts(), { providerReads: 0, credentialReads: 0 });
    });
  }
  for (const status of [404, 422, 501]) {
    test(endpoint + ' preserves native rejection ' + status + ' without retry', async () => {
      const error = { code: 'fixture_metadata_unavailable', message: 'Unavailable' };
      const h = harness(() => Response.json({ error }, { status }));
      const result = await h.call(endpoint, { model_id: artifact });
      assert.equal(result.ok, false);
      assert.equal(result.error.code, 'tyloinfer/core-rejected');
      assert.equal(result.error.details.http_status, status);
      assert.deepEqual(result.error.details.core_error, error);
      assert.equal(result.error.details.retryable, false);
      assert.equal(h.calls.length, 2);
    });
  }
  test(endpoint + ' uses Host credentials and rejects reflected credentials', async () => {
    const credential = 'fixture-protected-value';
    const h = harness((_url, init) => {
      assert.equal(init.headers.Authorization, 'Bearer ' + credential);
      return Response.json({ echo: credential });
    }, { provider: { apiKeyEnv: 'FIXTURE_CORE_CREDENTIAL' }, credential: { value: credential } });
    const result = await h.call(endpoint, { model_id: artifact });
    assert.equal(result.error.code, 'tyloinfer/core-invalid-response');
    assert.ok(!JSON.stringify(result).includes(credential));
  });
  test(endpoint + ' refuses incompatible core identity before reading metadata', async () => {
    const h = harness(() => assert.fail('metadata'), { health: { service: 'different-service', protocol_version: '1.0' } });
    assert.equal((await h.call(endpoint, { model_id: artifact })).error.code, 'tyloinfer/core-incompatible');
    assert.equal(h.calls.length, 1);
  });
  test(endpoint + ' honors an already cancelled read without network', async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call(endpoint, { model_id: artifact }, owner, AbortSignal.abort())).error.code, 'tyloinfer/cancelled');
    assert.equal(h.calls.length, 0);
  });
  test(endpoint + ' enforces the shared response limit without truncation', async () => {
    const h = harness(() => Response.json({ content: 'x'.repeat(512) }), { maxResponseBytes: 128 });
    assert.equal((await h.call(endpoint, { model_id: artifact })).error.code, 'tyloinfer/core-invalid-response');
    assert.equal(h.calls.length, 2);
  });
  test(endpoint + ' does not follow a core redirect', async () => {
    const h = harness((_url, init) => {
      assert.equal(init.redirect, 'error');
      return Response.json({ error: { code: 'redirect' } }, { status: 302, headers: { location: 'http://untrusted.invalid/' } });
    });
    const result = await h.call(endpoint, { model_id: artifact });
    assert.equal(result.error.code, 'tyloinfer/core-rejected');
    assert.equal(result.error.details.http_status, 302);
    assert.equal(h.calls.length, 2);
  });
}
for (const include_recurrent of [true, false]) {
  test('cache profile preserves an explicit include_recurrent=' + include_recurrent, async () => {
    const h = harness(url => { assert.equal(url.search, '?include_recurrent=' + include_recurrent); return Response.json(profile); });
    assert.deepEqual(await h.call('models/cache-profile', { model_id: artifact, include_recurrent }), { ok: true, value: profile });
  });
}
for (const include_recurrent of [null, 0, 1, 'false', 'true', {}, []]) {
  test('cache profile rejects non-boolean recurrent flag: ' + JSON.stringify(include_recurrent), async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call('models/cache-profile', { model_id: artifact, include_recurrent })).error.code, 'tyloinfer/bad-request');
    assert.equal(h.calls.length, 0);
    assert.deepEqual(h.counts(), { providerReads: 0, credentialReads: 0 });
  });
}
test('model directory rejects even a valid recurrent flag', async () => {
  const h = harness(() => assert.fail('network'));
  assert.equal((await h.call('models/directory', { model_id: artifact, include_recurrent: false })).error.code, 'tyloinfer/bad-request');
  assert.equal(h.calls.length, 0);
});
test('cache profile preserves native null rather than inventing capacity', async () => {
  const h = harness(() => Response.json(null));
  assert.deepEqual(await h.call('models/cache-profile', { model_id: artifact }), { ok: true, value: null });
});
test('model directory preserves native file metadata and capabilities', async () => {
  const h = harness(() => Response.json(directory));
  assert.deepEqual(await h.call('models/directory', { model_id: artifact }), { ok: true, value: directory });
  assert.ok(h.calls.every(call => call.method === 'GET'));
});
