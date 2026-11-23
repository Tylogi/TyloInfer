import { useSharedDisplay } from '../../presentation';
import type { ReactNode } from 'react';
import type { RuntimeRequestMetrics } from '@shared/types/api';
import { formatNumber } from '@shared/formatters';
import { Icon } from '@shared/ui/display';
import { displayPrefillMetric, preferPositiveMetric } from '@shared/features/runtime/metrics';

export function LogDetailsView({ value, children, onCopy }: { value: unknown; children?: ReactNode; onCopy?: () => void }) {
  const { tr,  } = useSharedDisplay();
  return <div className="log-record-detail">
    {children}
    <div className="log-detail-heading"><span>{tr('原始记录', 'Raw record')}</span>
      {onCopy && <button type="button" onClick={onCopy}><Icon name="copy" size={13} />{tr('复制', 'Copy')}</button>}</div>
    <pre>{JSON.stringify(value, null, 2)}</pre>
  </div>;
}

export function RequestMetricsView({ request }: { request: RuntimeRequestMetrics }) {
  const { tr,  } = useSharedDisplay();
  const prefill = displayPrefillMetric(request);
  const metric = (value: number | undefined, suffix = '') => value == null ? '--' : `${formatNumber(value, 1)}${suffix}`;
  const values = [
    [tr('输入 / 输出 tokens', 'Input / output tokens'), `${metric(request.prompt_tokens)} / ${metric(request.completion_tokens)}`],
    [tr('预填充', 'Prefill'), `${metric(prefill.tokensPerSecond, ' tok/s')} · ${metric(prefill.milliseconds, ' ms')}`],
    [tr('解码', 'Decode'), `${metric(request.decode_tps, ' tok/s')} · ${metric(request.decode_ms, ' ms')}`],
    [tr('首字延迟', 'TTFT'), metric(request.ttft_ms, ' ms')],
    [tr('请求总耗时', 'Total request time'), metric(preferPositiveMetric(request.complete_generation_ms, request.generation_ms), ' ms')],
    [tr('停止原因', 'Finish reason'), request.finish_reason || '--'],
  ];
  return <div className="log-detail-metrics">{values.map(([label, value]) =>
    <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>;
}

type DisplayEvent = { sequence: number; created_at: string; message?: string | null; type?: string; job_id?: string; data?: unknown };
export function JobEventsView({ events, loading, error }: { events: DisplayEvent[]; loading: boolean; error: string }) {
  const { tr,  } = useSharedDisplay();
  return <>
    {error && <p className="logs-error" role="status">{error}</p>}
    <div className="log-job-events">{events.length ? events.map(entry => <div key={`${entry.job_id || ''}:${entry.sequence}`}>
      <time>{new Date(entry.created_at).toLocaleTimeString()}</time>
      <p>{entry.type && <strong>{entry.type}{entry.message ? ' · ' : ''}</strong>}{entry.message || (!entry.type && '—')}</p>
      {entry.data != null && <details><summary>{tr('事件数据', 'Event data')}</summary><pre>{JSON.stringify(entry, null, 2)}</pre></details>}
    </div>) : <p>{loading ? tr('读取任务事件…', 'Loading job events…') : error
      ? tr('任务事件读取失败', 'Job events unavailable') : tr('没有附加事件', 'No additional events')}</p>}</div>
  </>;
}
