import test from 'node:test';
import assert from 'node:assert/strict';
import { mergeJobEvents } from '../job-events.ts';
const id = '11111111-1111-4111-8111-111111111111';
const event = (sequence: number) => ({ job_id: id, sequence, type: 'progress', level: 'info', message: null, data: { progress: 0.5 }, created_at: '2026-01-01T00:00:00Z' });

test('merges forward pages by job and sequence without dropping message-free native events', () => {
  const first = event(1), second = event(2), updated = { ...second, data: { progress: 0.75 } };
  const value = mergeJobEvents(id, [second, first], { data: [event(3), updated] });
  assert.deepEqual(value.map(item => item.sequence), [1, 2, 3]);
  assert.deepEqual(value[1], updated);
  assert.equal(value[2].message, null);
  assert.equal(value[2].job_id, id);
  assert.equal(value[2].type, 'progress');
});

test('rejects malformed pages and mismatched job identity rather than showing another job history', () => {
  for (const value of [null, {}, { data: null }, { data: [{ ...event(1), job_id: '22222222-2222-4222-8222-222222222222' }] }, { data: [{ ...event(1), sequence: -1 }] }, { data: [{ ...event(1), type: undefined }] }]) {
    assert.throws(() => mergeJobEvents(id, [], value));
  }
});
