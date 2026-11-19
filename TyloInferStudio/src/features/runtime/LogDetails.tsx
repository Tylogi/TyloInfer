import { useEffect, useState, type ReactNode } from 'react';
import { jobsApi } from '../../shared/api/resources/jobs';
import type { JobResource, RuntimeLogEntry } from '../../shared/api/types';
import { errorMessage } from '../../app/formatters';
import { JobEventsView, LogDetailsView, RequestMetricsView } from './LogDetailsView';
import { Icon } from '../../app/display';
import { useSettings } from '../settings/SettingsProvider';
import { toast } from '../../stores/toastStore';
import type { RequestLogRecord } from './useLogHistory';

export function LogDetails({ value, children }: { value: unknown; children?: ReactNode }) {
  const { tr } = useSettings();
  const serialized = JSON.stringify(value, null, 2);
  async function copy() {
    try { await navigator.clipboard.writeText(serialized); toast.success(tr('已复制', 'Copied')); }
    catch (cause) { toast.error(errorMessage(cause)); }
  }
  return <LogDetailsView value={value} onCopy={() => void copy()}>{children}</LogDetailsView>;
}

export function RequestDetails({ record }: { record: RequestLogRecord }) {
  return <LogDetails value={record}><RequestMetricsView request={record.request} /></LogDetails>;
}

export function JobDetails({ job, onRemove, busy, refreshAt }: { job: JobResource; onRemove: () => void; busy: boolean; refreshAt?: number | null }) {
  const { tr } = useSettings();
  const [logs, setLogs] = useState<RuntimeLogEntry[]>([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  useEffect(() => { setLogs([]); }, [job.id]);
  useEffect(() => {
    let disposed = false;
    const controller = new AbortController();
    setLoading(true); setError('');
    void jobsApi.jobEvents(job.id, controller.signal).then((next) => { if (!disposed) setLogs(next); })
      .catch((cause) => { if (!disposed) setError(errorMessage(cause)); })
      .finally(() => { if (!disposed) setLoading(false); });
    return () => { disposed = true; controller.abort(); };
  }, [job.id, job.updated_at, refreshAt]);
  const completed = ['succeeded', 'failed', 'cancelled', 'interrupted'].includes(job.status);
  return <LogDetails value={job}>
    {job.error && <p className="logs-error" role="status">{job.error.message}</p>}
    <div className="log-detail-heading"><span>{tr('任务事件', 'Job events')}</span>
      {completed && <button type="button" className="danger" disabled={busy} onClick={onRemove}>
        <Icon name="trash" size={13} />{tr('删除此记录', 'Delete this record')}</button>}</div>
    <JobEventsView events={logs} loading={loading} error={error} />
  </LogDetails>;
}
