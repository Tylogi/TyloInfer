import assert from 'node:assert/strict';
import test from 'node:test';
import { createCoreBridge } from '../../../lib/host/core-bridge.js';

const id = '11111111-1111-4111-8111-111111111111';
const other = '22222222-2222-4222-8222-222222222222';
const artifact = '1234567890abcdef1234567890abcdef';
const token = 'a'.repeat(64);
const owner = { principal: { kind: 'owner' } };
const source = { provider: 'huggingface', repo_id: 'example/model' };
const context = { instance_id: id, context_size: null };
const health = { service: 'tyloinfer-server', protocol_version: '1.0' };
const job = (kind = 'download.huggingface', status = 'failed', extra = {}) => ({ id, kind, status, cancel_requested: false, ...extra });
const plan = { model_id: artifact, name: 'fixture', files: ['fixture.tq'], total_bytes: 128, token };
const yarn = { supported: false, native_context: null, maximum_context: null, maximum_factor: null, enabled: false, effective_factor: 1 };
const instance = { id, pinned: true, idle_ttl_seconds: null, model: 'fixture', state: 'ready' };
function harness(route, options = {}) {
  const calls = [];
  let providerReads = 0;
  let credentialReads = 0;
  const bridge = createCoreBridge({
    getProvider: () => { providerReads++; return { baseURL: 'http://core.invalid/prefix/v1', ...(options.provider ?? {}) }; },
    resolveCredential: async () => { credentialReads++; return options.credential; },
    fetchImpl: async (url, init) => {
      calls.push({ path: url.pathname, url, ...init });
      if (url.pathname.endsWith('/health')) return Response.json(options.health ?? health);
      if (url.pathname.endsWith('/jobs/kinds')) return Response.json({ data: options.kinds ?? [
        { kind: 'download.huggingface' }, { kind: 'download.modelscope' }, { kind: 'runtime.context.configure' },
      ] });
      if (url.pathname.endsWith('/hub/download-directory')) return Response.json({ actual_directory: options.directory ?? '/fixture/models' });
      return route(url, init);
    },
    ...(options.maxResponseBytes ? { maxResponseBytes: options.maxResponseBytes } : {}),
  });
  return { calls, counts: () => ({ providerReads, credentialReads }),
    call: (endpoint, payload, peer = owner, signal) => bridge(endpoint, payload, signal, peer) };
}
const writes = [
  ['instances/configure', { instance_id: id, request: { pinned: true }, confirmed: true }],
  ['context/configure', { request: context, confirmed: true }],
  ['models/delete', { model_id: artifact, token, confirmed: true }],
  ['downloads/retry', { operation_id: id, confirmed: true }],
  ['downloads/delete', { operation_id: id, confirmed: true }],
];
const reads = [
  ['hub/readme', { ...source, revision: 'main' }], ['hub/probe-sources', { sources: [source] }],
  ['hub/download-reuse', { request: { ...source, destination: 'models/fixture' } }],
  ['instances/yarn', { instance_id: id }], ['context/get', { operation_id: id }], ['models/deletion-plan', { model_id: artifact }],
];
for (const [endpoint, payload] of [...reads, ...writes]) {
  test(endpoint + ' requires owner before provider or credential access', async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call(endpoint, payload, { principal: { kind: 'viewer' } })).error.code, 'tyloinfer/forbidden');
    assert.deepEqual(h.counts(), { providerReads: 0, credentialReads: 0 });
    assert.equal(h.calls.length, 0);
  });
}
for (const [endpoint, payload] of writes) {
  for (const confirmed of [undefined, false, 'true', null, 1]) {
    test(endpoint + ' rejects missing or non-boolean confirmation: ' + String(confirmed), async () => {
      const request = { ...payload };
      if (confirmed === undefined) delete request.confirmed; else request.confirmed = confirmed;
      const h = harness(() => assert.fail('network'));
      assert.equal((await h.call(endpoint, request)).error.code, 'tyloinfer/bad-request');
      assert.deepEqual(h.counts(), { providerReads: 0, credentialReads: 0 });
      assert.equal(h.calls.length, 0);
    });
  }
}
for (const content of [null, '# Untrusted model description']) {
  test('readme preserves nullable external content and encodes revision', async () => {
    const h = harness((url, init) => {
      assert.equal(init.method, 'GET');
      assert.equal(url.pathname, '/prefix/api/v1/hub/models/huggingface/example/model/readme');
      assert.equal(url.searchParams.get('revision'), 'refs/test & candidate');
      return Response.json({ content });
    });
    assert.deepEqual(await h.call('hub/readme', { ...source, revision: 'refs/test & candidate' }), { ok: true, value: { content } });
  });
}
test('source probe is a bounded read-only POST with native zero speed', async () => {
  const sources = [source, { provider: 'modelscope', repo_id: 'example/mirror', revision: null }];
  const value = { data: sources.map(row => ({ provider: row.provider, repo_id: row.repo_id, bytes_per_second: 0 })) };
  const h = harness((url, init) => {
    assert.equal(url.pathname, '/prefix/api/v1/hub/download-sources/probe');
    assert.equal(init.method, 'POST');
    assert.deepEqual(JSON.parse(init.body), { sources });
    return Response.json(value);
  });
  assert.deepEqual(await h.call('hub/probe-sources', { sources }), { ok: true, value });
  assert.equal(h.calls.length, 2);
});
for (const download_directory of [undefined, null, '/fixture/models']) {
  test('reuse preserves directory omission or the checked current directory: ' + String(download_directory), async () => {
    const request = { ...source, destination: 'models/fixture', ...(download_directory === undefined ? {} : { download_directory }) };
    const h = harness((url, init) => {
      assert.equal(url.pathname, '/prefix/api/v1/hub/download-reuse');
      assert.deepEqual(JSON.parse(init.body), request);
      return Response.json({ reusable_files: { 'fixture.tq': 128 } });
    });
    assert.deepEqual(await h.call('hub/download-reuse', { request }), { ok: true, value: { reusable_files: { 'fixture.tq': 128 } } });
    assert.equal(h.calls.length, typeof download_directory === 'string' ? 3 : 2);
  });
}
test('reuse refuses stale or arbitrary explicit directories before the external POST', async () => {
  const h = harness(() => assert.fail('external request'));
  assert.equal((await h.call('hub/download-reuse', { request: { ...source, destination: 'fixture', download_directory: '/outside' } })).error.code, 'tyloinfer/bad-request');
  assert.equal(h.calls.filter(call => call.method !== 'GET').length, 0);
});
test('instance policy uses PATCH, strips confirmation and preserves exact partial fields', async () => {
  const h = harness((url, init) => {
    assert.equal(url.pathname, '/prefix/api/v1/runtime/instances/' + id);
    assert.equal(init.method, 'PATCH');
    assert.deepEqual(JSON.parse(init.body), { pinned: true });
    return Response.json(instance);
  });
  assert.deepEqual(await h.call('instances/configure', { instance_id: id, request: { pinned: true }, confirmed: true }), { ok: true, value: instance });
});
test('YaRN reads native unsupported state without enabling or reloading', async () => {
  const h = harness((url, init) => {
    assert.equal(url.pathname, '/prefix/api/v1/runtime/yarn/' + id);
    assert.equal(init.method, 'GET');
    return Response.json(yarn);
  });
  assert.deepEqual(await h.call('instances/yarn', { instance_id: id }), { ok: true, value: yarn });
});
for (const kv_memory of [null, {}, { qsa_mode: 'auto', placement: 'auto_ssd', allocation: 'dynamic', ram_preallocated: false, ram_budget_bytes: 0, gpu_budget_bytes: null },
  { qsa_mode: 'fp16', placement: 'ram' }, { qsa_mode: 'affine8' }, { qsa_mode: 'tq4' }, { qsa_mode: 'legacy' }]) {
  test('context accepts native partial settings without inventing defaults: ' + JSON.stringify(kv_memory), async () => {
    const request = { ...context, kv_memory, yarn_enabled: null, qsa_kv_offload: { enabled: false, budget_bytes: 1, ram_budget_bytes: null }, kv_quantization: { bits: 2.5 } };
    const accepted = job('runtime.context.configure', 'queued');
    const h = harness((url, init) => {
      assert.equal(url.pathname, '/prefix/api/v1/runtime/context');
      assert.deepEqual(JSON.parse(init.body), request);
      return Response.json(accepted, { status: 202 });
    });
    assert.deepEqual(await h.call('context/configure', { request, confirmed: true }), { ok: true, value: accepted });
  });
}
test('context tracking reads only the native context job kind', async () => {
  const value = job('runtime.context.configure', 'running');
  const h = harness(() => Response.json(value));
  assert.deepEqual(await h.call('context/get', { operation_id: id }), { ok: true, value });
  assert.ok(h.calls.every(call => call.method === 'GET'));
});
test('context tracker cannot expose another kind or a cancel endpoint', async () => {
  const h = harness(() => Response.json(job()));
  assert.equal((await h.call('context/get', { operation_id: id })).error.code, 'tyloinfer/operation-not-supported');
  const before = h.calls.length;
  assert.equal((await h.call('context/cancel', { operation_id: id, confirmed: true })).error.code, 'tyloinfer/bad-request');
  assert.equal(h.calls.length, before);
});
test('deletion plan preserves the exact native target, files, size and token', async () => {
  const h = harness((url, init) => { assert.equal(init.method, 'GET'); assert.ok(url.pathname.endsWith('/' + artifact + '/deletion')); return Response.json(plan); });
  assert.deepEqual(await h.call('models/deletion-plan', { model_id: artifact }), { ok: true, value: plan });
});
test('model delete sends only its confirmed native plan token', async () => {
  const h = harness((url, init) => {
    assert.equal(url.pathname, '/prefix/api/v1/models/' + artifact + '/delete');
    assert.equal(init.method, 'POST'); assert.deepEqual(JSON.parse(init.body), { token });
    return Response.json({ files: 1, total_bytes: 128 });
  });
  assert.deepEqual(await h.call('models/delete', { model_id: artifact, token, confirmed: true }), { ok: true, value: { files: 1, total_bytes: 128 } });
});
for (const kind of ['download.huggingface', 'download.modelscope']) {
  test('download retry preserves actual returned identity for ' + kind, async () => {
    const h = harness((url, init) => Response.json(url.pathname.endsWith('/retry') ? job(kind, 'queued', { id: other }) : job(kind), { status: init.method === 'POST' ? 202 : 200 }));
    assert.equal((await h.call('downloads/retry', { operation_id: id, confirmed: true })).value.id, other);
    assert.equal(h.calls.filter(call => call.method === 'POST').length, 1);
  });
  test('download archive maps native 204 to null for ' + kind, async () => {
    const h = harness((_url, init) => init.method === 'DELETE' ? new Response(null, { status: 204 }) : Response.json(job(kind, 'succeeded')));
    assert.deepEqual(await h.call('downloads/delete', { operation_id: id, confirmed: true }), { ok: true, value: null });
    assert.equal(h.calls.filter(call => call.method === 'DELETE').length, 1);
  });
}
for (const endpoint of ['downloads/retry', 'downloads/delete']) {
  for (const state of ['queued', 'running', 'cancelling']) {
    test(endpoint + ' refuses active state without cancelling or writing: ' + state, async () => {
      const h = harness((_url, init) => { assert.equal(init.method, 'GET'); return Response.json(job('download.huggingface', state)); });
      assert.equal((await h.call(endpoint, { operation_id: id, confirmed: true })).error.code,
        endpoint === 'downloads/retry' ? 'tyloinfer/operation-not-retryable' : 'tyloinfer/operation-not-deletable');
    });
  }
  test(endpoint + ' refuses unrelated job kinds', async () => {
    const h = harness((_url, init) => { assert.equal(init.method, 'GET'); return Response.json(job('model.load')); });
    assert.equal((await h.call(endpoint, { operation_id: id, confirmed: true })).error.code, 'tyloinfer/operation-not-supported');
  });
}
test('retry refuses a missing download handler before POST', async () => {
  const h = harness((_url, init) => { assert.equal(init.method, 'GET'); return Response.json(job()); }, { kinds: [] });
  assert.equal((await h.call('downloads/retry', { operation_id: id, confirmed: true })).error.code, 'tyloinfer/operation-not-supported');
});
for (const [endpoint, payload] of [
  ['hub/readme', { ...source, revision: 'main', url: '/outside' }], ['hub/readme', source],
  ['hub/readme', { ...source, repo_id: '../model', revision: 'main' }], ['hub/probe-sources', { sources: [] }],
  ['hub/probe-sources', { sources: [source, source, source] }], ['hub/probe-sources', { sources: [{ ...source, provider: 'custom' }] }],
  ['hub/probe-sources', { sources: [{ ...source, url: 'http://outside.invalid' }] }],
  ...['../outside', '.', '/outside', 'X:\\outside', 'https://outside.invalid', 'folder/../outside', 'folder/./file', 'file:stream'].map(destination =>
    ['hub/download-reuse', { request: { ...source, destination } }]),
  ['instances/configure', { instance_id: id, request: {}, confirmed: true }],
  ['instances/configure', { instance_id: id, request: { pinned: null }, confirmed: true }],
  ['instances/configure', { instance_id: id, request: { idle_ttl_seconds: -1 }, confirmed: true }],
  ['instances/yarn', { instance_id: artifact }], ['instances/yarn', { instance_id: id, path: '/outside' }],
  ['context/configure', { request: { context_size: 512 }, confirmed: true }],
  ['context/configure', { request: { instance_id: id }, confirmed: true }],
  ['context/configure', { request: { ...context, apply_global_cap: true }, confirmed: true }],
  ['context/configure', { request: { ...context, context_size: 511 }, confirmed: true }],
  ...[{ qsa_kv_offload: { target_context: 512 } }, { qsa_kv_offload: { budget_bytes: 0 } }, { qsa_kv_offload: { ram_budget_bytes: 2 ** 50 + 1 } },
    { kv_quantization: { bits: 5 } }, { kv_quantization: { enabled: 1 } }, { kv_memory: { qsa_mode: 'automatic' } },
    { kv_memory: { ram_preallocated: false } }, { kv_memory: { ram_preallocated: false, placement: 'ram' } }, { kv_memory: { gpu_budget_bytes: 0 } }].map(settings =>
    ['context/configure', { request: { ...context, ...settings }, confirmed: true }]),
  ['models/deletion-plan', { model_id: id }], ['models/delete', { model_id: artifact, token: 'a'.repeat(63), confirmed: true }],
  ['models/delete', { model_id: artifact, token, confirmed: true, force: true }],
  ['downloads/retry', { operation_id: id, confirmed: true, request: {} }], ['downloads/delete', { operation_id: '../outside', confirmed: true }],
  ['jobs/delete', { job_id: id, confirmed: true, force: true }], ['jobs/retry', { job_id: id, confirmed: true }],
]) {
  test('rejects unsupported model action before provider: ' + endpoint + ' ' + JSON.stringify(payload), async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call(endpoint, payload)).error.code, 'tyloinfer/bad-request');
    assert.deepEqual(h.counts(), { providerReads: 0, credentialReads: 0 });
    assert.equal(h.calls.length, 0);
  });
}
for (const [endpoint, payload] of writes) {
  for (const failure of ['transport', 'server', 'invalid-json']) {
    test(endpoint + ' treats submitted write failure as unknown without retry: ' + failure, async () => {
      const h = harness((_url, init) => {
        if (init.method === 'GET') return Response.json(job());
        if (failure === 'transport') throw new Error('fixture transport failure');
        return failure === 'server' ? Response.json({ error: { code: 'fixture_failure' } }, { status: 503 }) : new Response('not-json');
      });
      const result = await h.call(endpoint, payload);
      assert.equal(result.error.code, 'tyloinfer/outcome-unknown');
      assert.equal(result.error.details.retryable, false);
      assert.equal(h.calls.filter(call => !['GET', 'HEAD'].includes(call.method)).length, 1);
    });
  }
}
for (const [endpoint, payload] of reads.filter(([endpoint]) => ['hub/probe-sources', 'hub/download-reuse'].includes(endpoint))) {
  for (const failure of ['transport', 'server', 'invalid-json']) {
    test(endpoint + ' never labels read-only POST failure as state-unknown: ' + failure, async () => {
      const h = harness(() => {
        if (failure === 'transport') throw new Error('fixture transport failure');
        return failure === 'server' ? Response.json({ error: { code: 'fixture_failure' } }, { status: 503 }) : new Response('not-json');
      });
      const result = await h.call(endpoint, payload);
      assert.equal(result.error.code, failure === 'transport' ? 'tyloinfer/core-unavailable' : failure === 'server' ? 'tyloinfer/core-rejected' : 'tyloinfer/core-invalid-response');
      assert.equal(h.calls.filter(call => call.method === 'POST').length, 1);
    });
  }
}
for (const [endpoint, payload] of [...reads, ...writes]) {
  test(endpoint + ' respects caller cancellation before dispatch', async () => {
    const h = harness(() => assert.fail('network'));
    assert.equal((await h.call(endpoint, payload, owner, AbortSignal.abort())).error.code, 'tyloinfer/cancelled');
    assert.equal(h.calls.length, 0);
  });
}
for (const [endpoint, payload, response] of [
  ['hub/readme', { ...source, revision: 'main' }, { content: 1 }],
  ['hub/probe-sources', { sources: [source] }, { data: [{ ...source, bytes_per_second: -1 }] }],
  ['hub/probe-sources', { sources: [source] }, { data: [{ ...source, repo_id: 'example/different', bytes_per_second: 1 }] }],
  ['hub/download-reuse', { request: { ...source, destination: 'fixture' } }, { reusable_files: { 'fixture.tq': -1 } }],
  ['instances/yarn', { instance_id: id }, { ...yarn, effective_factor: 0 }],
  ['models/deletion-plan', { model_id: artifact }, { ...plan, model_id: 'b'.repeat(32) }],
  ['models/deletion-plan', { model_id: artifact }, { ...plan, files: [] }],
]) {
  test(endpoint + ' rejects malformed native success instead of inventing empty data: ' + JSON.stringify(response), async () => {
    const h = harness(() => Response.json(response));
    assert.equal((await h.call(endpoint, payload)).error.code, 'tyloinfer/core-invalid-response');
  });
}
for (const [endpoint, payload, response, status] of [
  ['instances/configure', writes[0][1], { ...instance, id: other }, 200],
  ['context/configure', writes[1][1], job('model.load', 'queued'), 202],
  ['models/delete', writes[2][1], { files: -1, total_bytes: 128 }, 200],
  ['downloads/retry', writes[3][1], job('model.load', 'queued'), 202],
  ['downloads/delete', writes[4][1], {}, 200],
]) {
  test(endpoint + ' treats invalid write acceptance as unknown', async () => {
    const h = harness((_url, init) => init.method === 'GET' ? Response.json(job()) : Response.json(response, { status }));
    assert.equal((await h.call(endpoint, payload)).error.code, 'tyloinfer/outcome-unknown');
    assert.equal(h.calls.filter(call => call.method !== 'GET').length, 1);
  });
}
test('new read-only POSTs retain credential reflection protection', async () => {
  const credential = 'fixture-protected-core-value';
  const h = harness((_url, init) => {
    assert.equal(init.headers.Authorization, 'Bearer ' + credential);
    return Response.json({ reusable_files: {}, echo: credential });
  }, { provider: { apiKeyEnv: 'FIXTURE_CORE_CREDENTIAL' }, credential: { value: credential } });
  const result = await h.call('hub/download-reuse', { request: { ...source, destination: 'fixture' } });
  assert.equal(result.error.code, 'tyloinfer/core-invalid-response');
  assert.ok(!JSON.stringify(result).includes(credential));
});
