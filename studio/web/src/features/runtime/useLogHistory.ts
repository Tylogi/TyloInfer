import { useEffect, useRef, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { errorMessage } from '../../app/formatters';
import { runtimeApi } from '../../shared/api/resources/runtime';
import type { JobResource, RuntimeLogEntry } from '../../shared/api/types';
import { mergeRequestRecords, requestRecords, type RequestLogRecord } from './requestRecords';
export { mergeRequestRecords, requestRecords, type RequestLogRecord } from './requestRecords';
import { useJobStore } from '../../stores/jobStore';

export function useLogHistory(paused: boolean) {
  const { ready, connectionRevision } = useRuntime();
  const jobs = useJobStore((state) => state.jobs);
  const [logs, setLogs] = useState<RuntimeLogEntry[]>([]);
  const [requests, setRequests] = useState<RequestLogRecord[]>([]);
  const [jobSnapshot, setJobSnapshot] = useState<JobResource[]>(jobs);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const manual = useRef(0);
  const cursor = useRef<{ log?: number; metrics?: string }>({});
  useEffect(() => {
    cursor.current = {};
    setLogs([]); setRequests([]); setJobSnapshot([]); setError(''); setUpdatedAt(null);
  }, [ready, connectionRevision]);
  useEffect(() => { if (!paused && ready) setJobSnapshot(jobs); }, [jobs, paused, ready]);
  useEffect(() => {
    const forced = manual.current !== refreshVersion;
    manual.current = refreshVersion;
    if (!ready || (paused && !forced)) { setLoading(false); return; }
    let disposed = false;
    let polling = false;
    let manualPending = forced;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    async function poll() {
      if (disposed || polling) return;
      clearTimeout(timer);
      if (document.hidden) {
        if (!paused) timer = setTimeout(() => void poll(), 4000);
        return;
      }
      polling = true;
      manualPending = false;
      setLoading(true);
      try {
        const after = cursor.current.log;
        const result = await Promise.allSettled([
          runtimeApi.runtimeLogs(500, controller.signal, after).then((entries) =>
            !disposed && after != null && entries.length === 500
              ? runtimeApi.runtimeLogs(500, controller.signal) : entries),
          runtimeApi.runtimeMetrics(500, controller.signal, cursor.current.metrics),
        ]);
        if (disposed) return;
        const [events, metrics] = result;
        if (events.status === 'fulfilled' && events.value.length) {
          cursor.current.log = Math.max(...events.value.map((entry) => entry.sequence));
          setLogs((previous) => [...new Map([...previous, ...events.value].map((entry) => [entry.sequence, entry])).values()]
            .sort((a, b) => b.sequence - a.sequence).slice(0, 500));
        }
        if (metrics.status === 'fulfilled' && metrics.value.length) {
          cursor.current.metrics = [...metrics.value].sort((a, b) => b.sequence - a.sequence)[0].captured_at;
          const next = requestRecords(metrics.value);
          setRequests((previous) => mergeRequestRecords([...previous, ...next]).slice(0, 500));
        }
        setJobSnapshot(useJobStore.getState().jobs);
        setError(result.filter((item) => item.status === 'rejected').map((item) => errorMessage(item.reason)).join(' · '));
        if (result.some((item) => item.status === 'fulfilled')) setUpdatedAt(Date.now());
      } finally {
        polling = false;
        if (!disposed) {
          setLoading(false);
          if (!paused) timer = setTimeout(() => void poll(), 4000);
        }
      }
    }
    function visible() { if (!document.hidden && (!paused || manualPending)) void poll(); }
    document.addEventListener('visibilitychange', visible);
    void poll();
    return () => { disposed = true; controller.abort(); clearTimeout(timer); document.removeEventListener('visibilitychange', visible); };
  }, [ready, connectionRevision, paused, refreshVersion]);
  return { logs, requests, jobs: jobSnapshot, loading, error, updatedAt, ready,
    refresh: () => setRefreshVersion((version) => version + 1) };
}
