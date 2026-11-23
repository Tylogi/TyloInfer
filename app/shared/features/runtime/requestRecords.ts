import type { RuntimeMetricSnapshot, RuntimeRequestMetrics } from '@shared/types/api';

export interface RequestLogRecord {
  key: string;
  instance_id?: string | null;
  model?: string | null;
  captured_at: string;
  request: RuntimeRequestMetrics;
}

export function mergeRequestRecords(records: RequestLogRecord[]): RequestLogRecord[] {
  const merged = new Map<string, RequestLogRecord>();
  for (const record of records) {
    const previous = merged.get(record.key);
    const request = { ...previous?.request, ...record.request };
    if (previous?.request.completed_at != null && record.request.completed_at != null)
      request.completed_at = Math.max(previous.request.completed_at, record.request.completed_at);
    merged.set(record.key, { ...record, model: record.model ?? previous?.model, request });
  }
  return [...merged.values()].sort((a, b) =>
    (b.request.completed_at != null ? b.request.completed_at * 1000 : Date.parse(b.captured_at))
      - (a.request.completed_at != null ? a.request.completed_at * 1000 : Date.parse(a.captured_at)));
}

export function requestRecords(snapshots: RuntimeMetricSnapshot[]): RequestLogRecord[] {
  const records: RequestLogRecord[] = [];
  for (const snapshot of [...snapshots].sort((a, b) => a.sequence - b.sequence)) {
    const request = snapshot.values.last_request;
    if (!request?.id) continue;
    const key = `${snapshot.instance_id || ''}:${request.id}`;
    records.push({ key, instance_id: snapshot.instance_id, model: snapshot.model,
      captured_at: snapshot.captured_at, request });
  }
  return mergeRequestRecords(records);
}
