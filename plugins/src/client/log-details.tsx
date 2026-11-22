import React, { useEffect, useRef, useState } from 'react';
import { JobEventsView, LogDetailsView, RequestMetricsView } from '../../../studio/web/src/features/runtime/LogDetailsView.tsx';
import { CoreError } from './page-kit.tsx';
import { mergeJobEvents } from './job-events.ts';
import { terminalStates } from './model-api.ts';

export function ReadOnlyLogDetails({ value, request, children, tr }) {
  const [copyStatus, setCopyStatus] = useState('');
  const mounted = useRef(false);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  async function copy() {
    try { await navigator.clipboard.writeText(JSON.stringify(value, null, 2)); if (mounted.current) setCopyStatus(tr('已复制', 'Copied')); }
    catch { if (mounted.current) setCopyStatus(tr('无法复制，请手动选择记录文本。', 'Copy failed. Select the record text to copy manually.')); }
  }
  return <LogDetailsView value={value} onCopy={() => void copy()}>
    {request && <RequestMetricsView request={request} />}{children}
    {copyStatus && <p role="status">{copyStatus}</p>}
  </LogDetailsView>;
}

export function ReadOnlyJobDetails({ job, api, tr, onRemove, busy = false }) {
  const [state, setState] = useState({ job, events: [], loading: true, more: false, error: '', jobError: null });
  const request = useRef(null);
  const pending = useRef(true);
  useEffect(() => {
    const controller = new AbortController(); request.current = controller; pending.current = true;
    void Promise.allSettled([api.job(job.id, controller.signal), api.jobEvents(job.id, 0, controller.signal)]).then(([details, history]) => {
      if (controller.signal.aborted) return;
      const next = { job, events: [], loading: false, more: false, error: '', jobError: null };
      if (details.status === 'fulfilled' && details.value?.id === job.id) next.job = details.value;
      else next.jobError = details.status === 'rejected' ? details.reason : { message: tr('任务详情响应无效', 'Invalid job details response') };
      if (history.status === 'fulfilled') {
        try { next.events = mergeJobEvents(job.id, [], history.value); next.more = history.value.data.length === 200; }
        catch (error) { next.error = error.message; next.more = true; }
      } else { next.error = history.reason?.message || tr('任务事件读取失败', 'Job events unavailable'); next.more = true; }
      pending.current = false; setState(next);
    });
    return () => { controller.abort(); request.current = null; };
  }, [job.id, api]);
  async function more() {
    const controller = request.current;
    if (pending.current || !controller || controller.signal.aborted) return;
    pending.current = true; setState(current => ({ ...current, loading: true, error: '' }));
    const after = state.events.at(-1)?.sequence ?? 0;
    try {
      const value = await api.jobEvents(job.id, after, controller.signal);
      if (controller.signal.aborted) return;
      const events = mergeJobEvents(job.id, state.events, value);
      if (value.data.length && events.at(-1)?.sequence <= after) throw new Error('Job events cursor did not advance');
      setState(current => ({ ...current, events, more: value.data.length === 200, loading: false }));
    } catch (error) {
      if (!controller.signal.aborted) setState(current => ({ ...current, error: error.message, loading: false, more: true }));
    } finally { if (!controller.signal.aborted) pending.current = false; }
  }
  return <ReadOnlyLogDetails value={state.job} tr={tr}>
    {state.jobError && <CoreError error={state.jobError} />}
    {state.job.error && <p className="logs-error" role="status">{state.job.error.message}</p>}
    <div className="log-detail-heading">{tr('任务事件', 'Job events')}</div>
    <JobEventsView events={state.events} loading={state.loading} error={state.error} />
    {state.more && <button type="button" disabled={state.loading} onClick={() => void more()}>{state.error ? tr('重试读取事件', 'Retry reading events') : tr('读取后续事件', 'Load more events')}</button>}
    {state.more && <p>{tr('当前仅显示已读取的事件，不是完整历史。', 'Only retrieved events are shown; this is not the complete history.')}</p>}
    {onRemove && terminalStates.has(state.job.status) && <button type="button" className="danger" disabled={busy || state.loading || Boolean(state.jobError)} onClick={() => onRemove(state.job)}>{tr('删除这条记录', 'Delete this record')}</button>}
  </ReadOnlyLogDetails>;
}
