import test from 'node:test';
import assert from 'node:assert/strict';
import { describeCoreRuntime } from '../core-runtime.ts';

test('reads inventory through the native Connection with the caller signal', async () => {
  const controller = new AbortController();
  const calls: any[][] = [];
  const connection = { rpc: { call: async (...args: any[]) => { calls.push(args); return { ok: true, value: { data: args[2].resource === 'models' ? [{ id: 'example' }] : [] } }; } } };
  assert.deepEqual(await describeCoreRuntime(connection, controller.signal), { models: [{ id: 'example' }], loadedModels: [], instances: [] });
  assert.deepEqual(calls.map(call => call.slice(0, 3)), [['/tyloinfer', 'core/read', { resource: 'models', params: { include_components: true } }], ['/tyloinfer', 'core/read', { resource: 'loaded_models' }], ['/tyloinfer', 'core/read', { resource: 'instances' }]]);
  assert.ok(calls.every(call => call[3] === controller.signal));
});
test('does not fabricate empty inventories after a core rejection', async () => {
  await assert.rejects(describeCoreRuntime({ rpc: { call: async () => ({ ok: false, error: { code: 'tyloinfer/core-unavailable', message: 'Core unavailable' } }) } }), /Core unavailable/);
});
test('rejects malformed successful inventory payloads', async () => {
  for (const data of [null, {}, 'invalid']) await assert.rejects(describeCoreRuntime({ rpc: { call: async () => ({ ok: true, value: { data } }) } }), /Invalid inference inventory/);
});
test('preserves native request cancellation instead of returning a successful empty list', async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(describeCoreRuntime({ rpc: { call: async (_prefix: string, _method: string, _payload: any, signal: AbortSignal) => { signal.throwIfAborted(); } } }, controller.signal), { name: 'AbortError' });
});
