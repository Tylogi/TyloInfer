import test from 'node:test';
import assert from 'node:assert/strict';
import { createModelApi } from '../model-api.ts';
import { createModelManager } from '../model-manager.ts';
import { createOperationCheckpoint } from '../operation-checkpoint.ts';
const id = '11111111-1111-4111-8111-111111111111';
const instance = '22222222-2222-4222-8222-222222222222';
const ok = (value: any) => ({ ok: true, value });
const job = (status = 'running', rest: Record<string, any> = {}) => ({ id, kind: 'model.load', status, cancel_requested: false, progress: .25, ...rest });
const flush = () => new Promise(resolve => setImmediate(resolve));
function fixture(options: Record<string, any> = {}) {
  const calls: any[] = [];
  const timers = new Set();
  let loaded: any[] = [];
  let operation = job();
  let handler: any;
  const connection = { rpc: { call: async (channel: string, method: string, payload: any, signal: AbortSignal) => {
    calls.push({ channel, method, payload, signal });
    signal?.throwIfAborted();
    if (handler) { const reply = await handler(method, payload, signal); if (reply !== undefined) return reply; }
    if (method === 'core/read') return ok({ data: payload.resource === 'models' ? [{ id: 'artifact', name: 'Example', loadable: true }] : payload.resource === 'loaded_models' ? loaded : [] });
    if (method === 'models/load' || method === 'models/unload') return ok({ operation_id: id, status: 'accepted' });
    return ok(operation);
  } } };
  const manager = createModelManager(connection, { schedule: (callback: () => void) => { timers.add(callback); return () => timers.delete(callback); }, ...options });
  return { manager, connection, calls, timers, setLoaded: (value: any) => { loaded = value; }, setJob: (value: any) => { operation = value; }, intercept: (value: any) => { handler = value; },
    writes: () => calls.filter((call: any) => ['models/load', 'models/unload', 'operations/cancel'].includes(call.method)) };
}

test('passes native requests and signals once and never interprets accepted as complete', async () => {
  const controller = new AbortController();
  const calls: any[] = [];
  const api = createModelApi({ rpc: { call: async (...args) => { calls.push(args); return ok({ operation_id: id, status: 'accepted' }); } } });
  assert.deepEqual(await api.load({ model: 'artifact', context_size: 65536 }, controller.signal), { operation_id: id, status: 'accepted' });
  assert.deepEqual(calls[0], ['/tyloinfer', 'models/load', { request: { model: 'artifact', context_size: 65536 } }, controller.signal]);
  assert.equal(calls.length, 1);
});
test('treats write transport failure or malformed acceptance as unknown without retry', async () => {
  for (const response of [undefined, ok({ status: 'accepted' }), ok({ operation_id: id, status: 'succeeded' })]) {
    let count = 0;
    const api = createModelApi({ rpc: { call: async () => { count++; if (!response) throw new Error('transport'); return response; } } });
    await assert.rejects(api.load({ model: 'artifact' }), (error: any) => error.code === 'tyloinfer/outcome-unknown' && error.details.retryable === false);
    assert.equal(count, 1);
  }
});
test('retains structured core rejections and rejects non-model jobs', async () => {
  const rejection = { code: 'tyloinfer/core-rejected', message: 'Rejected', details: { http_status: 422, core_error: { code: 'invalid_context', message: 'Invalid context' } } };
  const api = createModelApi({ rpc: { call: async () => ({ ok: false, error: rejection }) } });
  await assert.rejects(api.unload({ instance_id: instance, force: false }), (error: any) => error.code === rejection.code && error.details.http_status === 422);
  await assert.rejects(createModelApi({ rpc: { call: async () => ok(job('running', { kind: 'download' })) } }).get(id), (error: any) => error.code === 'tyloinfer/core-invalid-response');
});
test('mount, refresh and navigation read three separate inventories without writes', async () => {
  const f = fixture();
  const stop = f.manager.watch();
  await flush();
  assert.deepEqual(f.calls.map((call: any) => call.payload.resource), ['models', 'loaded_models', 'instances']);
  assert.equal(f.manager.getSnapshot().inventory!.loadedModels.length, 0);
  await f.manager.refresh();
  stop();
  const stopAgain = f.manager.watch();
  await flush();
  assert.equal(f.writes().length, 0);
  stopAgain(); f.manager.dispose();
});
test('guards duplicate submits and refreshes actual inventory only after native terminal success', async () => {
  const f = fixture();
  await f.manager.refresh();
  let resolve: any;
  f.intercept((method: any) => method === 'models/load' ? new Promise(done => { resolve = done; }) : undefined);
  const submission = f.manager.load({ model: 'artifact', pin: false });
  assert.equal(await f.manager.load({ model: 'artifact' }), false);
  resolve(ok({ operation_id: id, status: 'accepted' }));
  assert.equal(await submission, true);
  assert.equal(f.manager.getSnapshot().operation!.status, 'accepted');
  assert.equal(f.manager.getSnapshot().inventory!.loadedModels.length, 0);
  const stop = f.manager.watch();
  await flush();
  assert.equal(f.manager.getSnapshot().operation!.status, 'running');
  assert.equal(f.timers.size, 1);
  f.setLoaded([{ id: 'loaded', model: 'artifact', instance_id: instance, state: 'ready' }]);
  f.setJob(job('succeeded', { progress: 1 }));
  await f.manager.poll();
  assert.equal(f.manager.getSnapshot().operation!.status, 'succeeded');
  assert.equal(f.manager.getSnapshot().inventory!.loadedModels[0].instance_id, instance);
  assert.equal(f.timers.size, 0);
  assert.equal(f.writes().length, 1);
  stop(); f.manager.dispose();
});
test('requires an actual runtime instance for unload and sends force false unless explicitly selected', async () => {
  const f = fixture();
  f.setLoaded([{ id: 'loaded', model: 'artifact', instance_id: instance, state: 'ready' }]);
  await f.manager.refresh();
  assert.equal(await f.manager.unload({ instance_id: 'artifact' }), false);
  assert.equal(await f.manager.unload({ instance_id: instance }), true);
  assert.deepEqual(f.writes()[0].payload, { request: { instance_id: instance, force: false } });
  f.manager.dispose();
});
test('cancellation is single-shot and cancelling is not cancelled', async () => {
  const f = fixture();
  const stop = f.manager.watch();
  await flush();
  await f.manager.load({ model: 'artifact' });
  await flush();
  f.setJob(job('cancelling', { cancel_requested: true }));
  assert.equal(await f.manager.cancel(), true);
  assert.equal(await f.manager.cancel(), false);
  assert.equal(f.manager.getSnapshot().operation!.status, 'cancelling');
  assert.equal(f.writes().filter((call: any) => call.method === 'operations/cancel').length, 1);
  f.setJob(job('cancelled', { cancel_requested: true }));
  await f.manager.poll();
  assert.equal(f.manager.getSnapshot().operation!.status, 'cancelled');
  stop(); f.manager.dispose();
});
test('unknown outcomes block further writes until a subsequent successful read and explicit acknowledgement', async () => {
  const f = fixture();
  await f.manager.refresh();
  f.intercept((method: any) => { if (method === 'models/load') throw new Error('transport'); });
  assert.equal(await f.manager.load({ model: 'artifact' }), false);
  assert.equal(f.manager.getSnapshot().uncertainty!.reviewed, false);
  f.manager.acknowledge();
  assert.equal(await f.manager.load({ model: 'artifact' }), false);
  assert.equal(f.writes().length, 1);
  await f.manager.refresh();
  assert.equal(f.manager.getSnapshot().uncertainty!.reviewed, true);
  assert.equal(await f.manager.load({ model: 'artifact' }), false);
  f.manager.acknowledge();
  assert.equal(f.manager.getSnapshot().uncertainty, null);
  f.manager.dispose();
});
test('preserves failed job state and does not fabricate a loaded model', async () => {
  const f = fixture();
  const stop = f.manager.watch();
  await flush();
  f.setJob(job('failed', { error: { message: 'Out of memory' } }));
  await f.manager.load({ model: 'artifact' });
  await flush();
  assert.equal(f.manager.getSnapshot().operation!.status, 'failed');
  assert.equal(f.manager.getSnapshot().operation!.error.message, 'Out of memory');
  assert.equal(f.manager.getSnapshot().inventory!.loadedModels.length, 0);
  assert.equal(f.writes().length, 1);
  stop(); f.manager.dispose();
});
test('navigation aborts reads and timers, retains tickets and never cancels the backend implicitly', async () => {
  const f = fixture();
  const stop = f.manager.watch();
  await flush();
  await f.manager.load({ model: 'artifact' });
  await flush();
  const writesBefore = f.writes().length;
  stop();
  assert.equal(f.timers.size, 0);
  assert.equal(f.manager.getSnapshot().operation!.id, id);
  assert.ok(f.calls.filter((call: any) => call.method === 'operations/get').every((call: any) => call.signal.aborted));
  const stopAgain = f.manager.watch();
  await flush();
  assert.equal(f.writes().length, writesBefore);
  stopAgain(); f.manager.dispose();
});
test('failed inventory reads disable writes rather than silently showing empty lists', async () => {
  const f = fixture();
  await f.manager.refresh();
  f.intercept((method: any) => method === 'core/read' ? ok({ data: null }) : undefined);
  await f.manager.refresh();
  assert.ok(f.manager.getSnapshot().inventoryError);
  assert.equal(f.manager.getSnapshot().inventory!.models.length, 1);
  assert.equal(await f.manager.load({ model: 'artifact' }), false);
  assert.equal(f.writes().length, 0);
  f.manager.dispose();
});
test('restoring a ticket performs only reads and blocks submissions while tracking is unresolved', async () => {
  const f = fixture();
  await f.manager.refresh();
  let resolve: any;
  f.intercept((method: any) => method === 'operations/get' ? new Promise(done => { resolve = done; }) : undefined);
  const tracking = f.manager.track(id);
  assert.equal(await f.manager.load({ model: 'artifact' }), false);
  resolve(ok(job()));
  assert.equal(await tracking, true);
  assert.equal(f.manager.getSnapshot().operation!.id, id);
  assert.equal(f.writes().length, 0);
  f.manager.dispose();
});
test('automatically resumes a saved model operation after refresh with reads only', async () => {
  const values = new Map();
  const storage = { getItem: (key: any) => values.get(key), setItem: (key: string, value: string) => values.set(key, value), removeItem: (key: any) => values.delete(key) };
  const checkpoint = createOperationCheckpoint(storage, 'model');
  const f = fixture({ checkpoint });
  await f.manager.refresh();
  await f.manager.load({ model: 'artifact' });
  assert.equal(checkpoint.load()!.id, id);
  f.manager.dispose();
  const recovered = createModelManager(f.connection, { checkpoint, schedule: () => () => {} });
  assert.equal(recovered.getSnapshot().tracking, true);
  const stop = recovered.watch();
  await flush();
  assert.equal(recovered.getSnapshot().operation!.status, 'running');
  assert.equal(recovered.getSnapshot().operationLabel, 'Example');
  assert.equal(f.writes().length, 1);
  f.setJob(job('succeeded'));
  await recovered.poll();
  assert.equal(checkpoint.load(), null);
  stop(); recovered.dispose();
});
test('refresh during an unresolved submission never replays the write', async () => {
  const values = new Map();
  const checkpoint = createOperationCheckpoint({ getItem: (key: any) => values.get(key), setItem: (key: string, value: string) => values.set(key, value), removeItem: (key: any) => values.delete(key) }, 'model');
  const f = fixture({ checkpoint });
  await f.manager.refresh();
  let release: any;
  f.intercept((method: any) => method === 'models/load' ? new Promise(resolve => { release = resolve; }) : undefined);
  const pending = f.manager.load({ model: 'artifact' });
  assert.equal(checkpoint.load()!.unknown, true);
  f.manager.dispose();
  const recovered = createModelManager(f.connection, { checkpoint, schedule: () => () => {} });
  await recovered.refresh();
  assert.ok(recovered.getSnapshot().uncertainty);
  assert.equal(await recovered.load({ model: 'artifact' }), false);
  assert.equal(f.writes().length, 1);
  release(ok({ operation_id: id, status: 'accepted' }));
  await pending;
  assert.equal(checkpoint.load()!.unknown, true);
  recovered.dispose();
});
test('ignores corrupt checkpoints and tolerates unavailable browser storage', () => {
  for (const text of ['invalid json', JSON.stringify({ version: 1, id: 'invalid', kind: 'model.load', unknown: false })]) {
    const checkpoint = createOperationCheckpoint({ getItem: () => text }, 'model');
    assert.equal(checkpoint.load(), null);
  }
  const unavailable = createOperationCheckpoint(() => { throw new Error('unavailable'); }, 'model');
  assert.equal(unavailable.load(), null);
  assert.doesNotThrow(() => unavailable.save({ id, kind: 'model.load' }));
});
test('plugin disposal aborts transmission without claiming background cancellation', async () => {
  const f = fixture();
  await f.manager.refresh();
  let resolve: any;
  f.intercept((method: any) => method === 'models/load' ? new Promise(done => { resolve = done; }) : undefined);
  const pending = f.manager.load({ model: 'artifact' });
  const before = f.manager.getSnapshot();
  f.manager.dispose();
  resolve(ok({ operation_id: id, status: 'accepted' }));
  await pending;
  assert.equal(f.manager.getSnapshot(), before);
  assert.equal(f.writes()[0].signal.aborted, true);
  assert.equal(f.writes().length, 1);
});
