import assert from 'node:assert/strict';
import { setTimeout } from 'node:timers/promises';
import { createCoreBridge } from '../core-bridge.js';

const origin = process.argv[2];
assert.equal(new URL(origin).hostname, '127.0.0.1');
const owner = { principal: { kind: 'owner' } };
const handler = createCoreBridge({ getProvider: () => ({ baseURL: origin + '/v1' }), resolveCredential: async () => undefined });
const call = (method, payload) => handler(method, payload, new AbortController().signal, owner);
const accepted = async (method, request) => {
  const reply = await call(method, { request });
  assert.equal(reply.ok, true, JSON.stringify(reply));
  assert.equal(reply.value.status, 'accepted');
  assert.match(reply.value.operation_id, /^[0-9a-f-]{36}$/);
  return reply.value.operation_id;
};
const waitFor = async (id, state) => {
  for (let attempt = 0; attempt < 100; attempt++) {
    const reply = await call('operations/get', { operation_id: id });
    assert.equal(reply.ok, true, JSON.stringify(reply));
    if (reply.value.status === state) return reply.value;
    assert.ok(!['failed', 'interrupted'].includes(reply.value.status));
    await setTimeout(10);
  }
  throw new Error('Controlled operation did not reach ' + state);
};
const success = await accepted('models/load', { model: 'bridge-fixture-success', context_size: 512, kv_memory: { qsa_mode: 'auto', placement: 'auto' } });
assert.equal((await waitFor(success, 'succeeded')).kind, 'model.load');
const pending = await accepted('models/load', { model: 'bridge-fixture-pending' });
await waitFor(pending, 'running');
const cancellation = await call('operations/cancel', { operation_id: pending });
assert.equal(cancellation.ok, true, JSON.stringify(cancellation));
assert.equal(cancellation.value.cancel_requested, true);
await waitFor(pending, 'cancelled');
const unloaded = await accepted('models/unload', { instance_id: '22222222-2222-4222-8222-222222222222', force: false });
assert.equal((await waitFor(unloaded, 'succeeded')).kind, 'model.unload');
const invalid = await call('models/load', { request: { model: 'bridge-fixture-invalid', context_size: 1 } });
assert.equal(invalid.error.code, 'tyloinfer/core-rejected');
assert.equal(invalid.error.details.http_status, 422);
console.log('CORE_MODEL_BRIDGE_API_PASS; RUNTIME=controlled; GPU=unused');
