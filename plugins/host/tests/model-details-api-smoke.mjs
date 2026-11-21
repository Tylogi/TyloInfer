import assert from 'node:assert/strict';
import { createCoreBridge } from '../core-bridge.js';

const [origin, known, unknown] = process.argv.slice(2);
assert.equal(new URL(origin).hostname, '127.0.0.1');
const calls = [];
const handler = createCoreBridge({
  getProvider: () => ({ baseURL: origin + '/v1' }),
  resolveCredential: async () => undefined,
  fetchImpl: (url, init) => {
    assert.equal(init.method, 'GET');
    calls.push(url.pathname);
    return fetch(url, init);
  },
});
const call = (endpoint, payload) => handler(endpoint, payload, undefined, { principal: { kind: 'owner' } });
const value = async (endpoint, payload) => {
  const result = await call(endpoint, payload);
  assert.equal(result.ok, true, JSON.stringify(result));
  return result.value;
};
const native = async path => {
  const response = await fetch(origin + path, { method: 'GET', redirect: 'error' });
  assert.equal(response.status, 200);
  return response.json();
};
const standard = await value('models/cache-profile', { model_id: known });
assert.deepEqual(standard, await native('/api/v1/models/' + known + '/cache-profile'));
assert.ok(standard.components.length > 0);
for (const include_recurrent of [false, true]) {
  const result = await value('models/cache-profile', { model_id: known, include_recurrent });
  assert.deepEqual(result, await native('/api/v1/models/' + known + '/cache-profile?include_recurrent=' + include_recurrent));
  if (include_recurrent) {
    assert.ok(result.fixed_bytes > standard.fixed_bytes);
    assert.ok(result.fixed_components.some(component => component.name === 'recurrent_state'));
  } else assert.deepEqual(result, standard);
}
assert.equal(await value('models/cache-profile', { model_id: unknown }), null);
for (const model_id of [known, unknown]) {
  const result = await value('models/directory', { model_id });
  assert.deepEqual(result, await native('/api/v1/models/' + model_id + '/directory'));
  assert.match(result.current_id, /^[0-9a-f]{32}$/);
  assert.equal(result.model_file_count, 2);
  assert.deepEqual(result.files.map(file => file.name), ['known.tq', 'unknown.tq']);
}
for (const endpoint of ['models/cache-profile', 'models/directory']) {
  const result = await call(endpoint, { model_id: '0'.repeat(32) });
  assert.equal(result.ok, false);
  assert.equal(result.error.code, 'tyloinfer/core-rejected');
  assert.equal(result.error.details.http_status, 404);
  assert.equal(result.error.details.core_error.code, 'model_artifact_not_found');
}
assert.ok(calls.every(path => path === '/health' || /^\/api\/v1\/models\/[0-9a-f]{32}\/(cache-profile|directory)$/.test(path)));
console.log('CORE_MODEL_DETAILS_BRIDGE_API_PASS; MODELS=isolated-fixtures; CORE_WRITES=0; TENSOR_DECODE=none; DOWNLOADS=none');
