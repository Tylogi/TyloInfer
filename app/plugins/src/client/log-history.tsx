import { useEffect, useRef, useState } from 'react';
import { mergeRequestRecords, requestRecords } from '@shared/features/runtime/requestRecords.ts';

const empty = () => ({ logs: [], requests: [], jobs: [], errors: {}, updatedAt: null });
export function useLogHistory(api, values, paused, refreshKey) {
  const [history, setHistory] = useState(empty);
  const cursor = useRef({});
  const current = useRef(history); current.current = history;
  const latestPaused = useRef(paused); latestPaused.current = paused;
  const initialRefresh = useRef(refreshKey);
  const appliedValues = useRef(null);
  function apply(entries) {
    setHistory(previous => {
      const next = { ...previous, errors: { ...previous.errors } }; let retrieved = false;
      for (const key of ['logs', 'metrics', 'jobs']) {
        const entry = entries[key]; if (!entry) continue;
        if (entry.error) { next.errors[key] = entry.error; continue; }
        if (!Array.isArray(entry.value?.data)) { next.errors[key] = { message: 'Invalid log history response' }; continue; }
        delete next.errors[key]; retrieved = true;
        const rows = entry.value.data;
        if (key === 'logs') {
          next.logs = [...new Map([...previous.logs, ...rows].map(item => [item.sequence, item])).values()].sort((a, b) => b.sequence - a.sequence).slice(0, 500);
          if (rows.length) cursor.current.log = Math.max(cursor.current.log ?? 0, ...rows.map(item => item.sequence));
        } else if (key === 'metrics') {
          next.requests = mergeRequestRecords([...previous.requests, ...requestRecords(rows)]).slice(0, 500);
          if (rows.length) cursor.current.metrics = [...rows].sort((a, b) => b.sequence - a.sequence)[0].captured_at;
        } else next.jobs = rows;
      }
      if (retrieved) next.updatedAt = Date.now();
      return next;
    });
  }
  useEffect(() => { cursor.current = {}; setHistory(empty()); }, [api]);
  useEffect(() => {
    const manual = initialRefresh.current !== refreshKey;
    if (!Object.keys(values).length || appliedValues.current === values && !manual) return;
    initialRefresh.current = refreshKey;
    if (!paused || manual || !current.current.updatedAt) { appliedValues.current = values; apply(values); }
  }, [values, refreshKey, paused]);
  useEffect(() => {
    if (paused) return;
    const controller = new AbortController(); let timer; let pending = false;
    async function poll() {
      if (pending || controller.signal.aborted || latestPaused.current) return;
      if (document.hidden) { timer = setTimeout(poll, 4000); return; }
      pending = true;
      const after = cursor.current.log;
      const since = cursor.current.metrics;
      const entries = await Promise.all(['logs', 'metrics', 'jobs'].map(async resource => {
        try {
          let value = await api.read(resource, resource === 'logs' ? { limit: 500, ...(after == null ? {} : { after }) } : resource === 'metrics' ? { limit: 500, ...(since ? { since } : {}) } : { limit: 200 }, controller.signal);
          if (resource === 'logs' && after != null && value.data?.length === 500) value = await api.read(resource, { limit: 500 }, controller.signal);
          return [resource, { value }];
        } catch (error) { return [resource, { error }]; }
      }));
      pending = false;
      if (!controller.signal.aborted && !latestPaused.current) { apply(Object.fromEntries(entries)); timer = setTimeout(poll, 4000); }
    }
    function visible() { if (!document.hidden) { clearTimeout(timer); void poll(); } }
    timer = setTimeout(poll, 4000); document.addEventListener('visibilitychange', visible);
    return () => { controller.abort(); clearTimeout(timer); document.removeEventListener('visibilitychange', visible); };
  }, [api, paused, refreshKey]);
  return history;
}
