import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultLoadOptions, nativeLoadRequest, normalizeLoadOptions, loadModelDrafts, saveModelDrafts } from '../src/model-load-options.js';

test('asks the core to choose context and precision using the default reserve', () => {
  assert.deepEqual(nativeLoadRequest({ id: 'a'.repeat(32) }, defaultLoadOptions()), {
    model: 'a'.repeat(32), pin: false, context_management: { reserve_percent: 10, precision: null, vram_preallocated: true },
  });
});
test('passes explicit context, reserve, KV, device and residency fields', () => {
  const options = defaultLoadOptions();
  options.context.context_size = 65536;
  options.context.reserve_percent = 20;
  options.context.precision = 4;
  options.context.vram_preallocated = false;
  options.devices = ['0']; options.ttl = 900;
  const request = nativeLoadRequest({ id: 'a'.repeat(32) }, options);
  assert.equal(request.context_size, 65536);
  assert.deepEqual(request.context_management, { reserve_percent: 20, precision: 4, vram_preallocated: false });
  assert.deepEqual(request.device_ids, ['0']);
  assert.equal(request.idle_ttl_seconds, 900);
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
