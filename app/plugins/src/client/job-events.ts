import type { JobEvent } from './types.ts';

export function mergeJobEvents(jobId: string, previous: JobEvent[], value: any): JobEvent[] {
  if (!Array.isArray(value?.data)) throw new Error('Invalid job events response');
  const events = new Map(previous.map((event: JobEvent) => [`${event.job_id}:${event.sequence}`, event]));
  for (const event of value.data) {
    if (event?.job_id !== jobId || !Number.isSafeInteger(event.sequence) || event.sequence < 0 || typeof event.type !== 'string' || typeof event.created_at !== 'string') throw new Error('Invalid job event');
    events.set(`${event.job_id}:${event.sequence}`, event);
  }
  return [...events.values()].sort((a, b) => a.sequence - b.sequence);
}
