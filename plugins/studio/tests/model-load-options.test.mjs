import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultLoadOptions, nativeLoadRequest, normalizeLoadOptions, loadModelDrafts, saveModelDrafts } from '../src/model-load-options.js';

test('leaves core defaults alone until the user changes next-load options', () => {
  assert.deepEqual(nativeLoadRequest({ id: 'a'.repeat(32) }, defaultLoadOptions()), { model: 'a'.repeat(32), pin: false });
});
test('uses original context, KV, device and residency fields without a second load path', () => {
  const options = defaultLoadOptions();
  options.contextChanged = true;
  options.context.context_size = 65536;
  options.context.automatic = false;
  options.context.ssd_kv = true;
  options.context.kv_quantization = { algorithm: 'turboquant', enabled: true, bits: 4 };
  options.devices = ['0']; options.ttl = 900; options.chunk = '2048';
  const request = nativeLoadRequest({ id: 'a'.repeat(32) }, options);
  assert.equal(request.context_size, 65536);
  assert.equal(request.context_management.ssd_kv, true);
  assert.equal(request.context_management.kv_quantization.bits, 4);
  assert.deepEqual(request.device_ids, ['0']);
  assert.equal(request.idle_ttl_seconds, 900);
  assert.equal(request.prefill_chunk_size, 2048);
  assert.ok(!('jev_weights' in request));
  options.pin = true;
  assert.ok(!('idle_ttl_seconds' in nativeLoadRequest({ id: 'a'.repeat(32) }, options)));
});
test('keeps browser drafts per artifact and discards malformed saved values', () => {
  const map = new Map();
  const storage = () => ({ getItem: key => map.get(key), setItem: (key, value) => map.set(key, value) });
  const options = defaultLoadOptions(); options.context.context_size = 8192;
  saveModelDrafts(storage, { ['a'.repeat(32)]: options });
  assert.equal(loadModelDrafts(storage)['a'.repeat(32)].context.context_size, 8192);
  assert.equal(loadModelDrafts(storage)['b'.repeat(32)], undefined);
  assert.deepEqual(normalizeLoadOptions({ pin: 'false', context: { context_size: -1, kv_quantization: { enabled: true, bits: 99 } } }), defaultLoadOptions());
  assert.deepEqual(loadModelDrafts(() => { throw new Error('blocked'); }), {});
});
