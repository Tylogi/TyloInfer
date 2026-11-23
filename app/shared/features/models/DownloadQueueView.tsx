import type { JobResource } from '@shared/types/api';
import { Icon } from '@shared/ui/display';
import { formatBytes, formatDuration, formatNumber } from '@shared/formatters';
import { ModelVendorMark } from '@shared/ui/ModelVendorMark';
import type { DownloadOrigin } from './ModelBrowserView';

export const isModelDownload = (job: JobResource) => job.kind === 'download.huggingface' || job.kind === 'download.modelscope';
export const isActiveDownload = (job: JobResource) => ['queued', 'running', 'cancelling'].includes(job.status);
export const downloadProgress = (job: JobResource) => Number.isFinite(job.progress) ? Math.max(0, Math.min(1, job.progress)) : 0;

export function downloadEtaSeconds(job: JobResource): number | null {
  const { downloaded_bytes: downloaded, total_bytes: total, bytes_per_second: speed } = job.progress_data ?? {};
  if (job.status !== 'running' || downloaded == null || total == null || speed == null ||
    !Number.isFinite(downloaded) || !Number.isFinite(total) || !Number.isFinite(speed) ||
    downloaded < 0 || total <= downloaded || speed < 1) return null;
  const seconds = (total - downloaded) / speed;
  return Number.isFinite(seconds) ? Math.ceil(seconds) : null;
}

export function DownloadQueueView({ jobs, busyId = null, allowRetry = true, canRetry, onAction, onRemove, onTrack, tr }: {
  jobs: JobResource[]; busyId?: string | null; allowRetry?: boolean; canRetry?(job: JobResource): boolean;
  onAction?(job: JobResource, retry: boolean, origin: DownloadOrigin): void;
  onRemove?(job: JobResource): void; onTrack?(job: JobResource): void; tr(zh: string, en: string): string;
}) {
  const statusLabels: Record<JobResource['status'], string> = {
    queued: tr('等待下载', 'Queued'), running: tr('下载中', 'Downloading'),
    cancelling: tr('正在取消', 'Cancelling'), succeeded: tr('下载完成', 'Completed'),
    failed: tr('下载失败', 'Failed'), cancelled: tr('已取消', 'Cancelled'),
    interrupted: tr('下载中断', 'Interrupted'),
  };
  return (
    <section className="download-queue" aria-label={tr('下载队列', 'Download queue')}>
      <div className="download-queue-heading">
        <h3>{tr('下载队列', 'Download queue')}</h3>
        <span>{tr(`${jobs.filter(isActiveDownload).length} 个下载中`, `${jobs.filter(isActiveDownload).length} active`)}</span>
      </div>
      {!jobs.length && <div className="download-queue-empty"><Icon name="download" size={28} />
        <strong>{tr('暂无下载', 'No downloads yet')}</strong>
      </div>}
      {jobs.map((job) => {
        const progress = downloadProgress(job);
        const eta = downloadEtaSeconds(job);
        const total = job.progress_data?.total_bytes;
        const downloaded = job.progress_data?.downloaded_bytes;
        const finishing = total != null && Number.isFinite(total) && total > 0 &&
          downloaded != null && Number.isFinite(downloaded) && downloaded >= total;
        const duration = eta == null ? '' : eta < 60 ? `${eta}s` : formatDuration(eta);
        const etaLabel = eta != null ? tr(`预计剩余 ${duration}`, `${duration} remaining`)
          : finishing ? tr('正在收尾', 'Finishing download') : tr('剩余时间估算中…', 'Estimating time remaining…');
        const retry = ['failed', 'cancelled', 'interrupted'].includes(job.status);
        const files = Array.isArray(job.payload.include)
          ? Array.from(new Set(job.payload.include.filter((file): file is string => typeof file === 'string' && !!file && !/[?*\[]/.test(file))))
          : [];
        const destination = String(job.payload.destination || '');
        const primary = Array.isArray(job.payload.include) ? job.payload.include.find(
          (file): file is string => typeof file === 'string' && /\.tq$/i.test(file) && !/(?:^|\/)Engram\//i.test(file)) : undefined;
        const title = primary || files[0] || destination.split(/[\\/]/).filter(Boolean).pop() || String(job.payload.repo_id || job.kind);
        return (
          <article className={`download-queue-item ${job.status}`} key={job.id} data-job-id={job.id}>
            <div className="download-item-icon"><Icon name={job.status === 'succeeded' ? 'check' : 'download'} size={20} /></div>
            <div className="download-item-body">
              <div className="download-item-heading"><strong>{title}</strong><span className="model-identity-trailing">{statusLabels[job.status]}<ModelVendorMark name={String(job.payload.repo_id || '')} /></span></div>
              <small><span>{String(job.payload.repo_id || job.kind)}</span> · {job.kind === 'download.huggingface' ? 'Hugging Face' : 'ModelScope'}</small>
              {files.length > 1 && <details className="download-item-files">
                <summary>{tr(`查看 ${files.length} 个文件`, `View ${files.length} files`)}</summary>
                <ul>{files.map((file) => <li key={file}>{file}</li>)}</ul>
              </details>}
              <small>{tr('保存到', 'Save to')} · {destination}</small>
              <div className="download-item-progress"><progress aria-label={tr('下载进度', 'Download progress')} max={1} value={progress} /><span>{formatNumber(progress * 100, 1)}%</span></div>
              {isActiveDownload(job) && job.progress_data?.downloaded_bytes != null && <small className="download-transfer-stats">
                {(job.progress_data.bytes_per_second ? formatBytes(job.progress_data.bytes_per_second).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`) : '0 B')}/s
                {' · '}{formatBytes(job.progress_data.downloaded_bytes)}{job.progress_data.total_bytes ? ` / ${formatBytes(job.progress_data.total_bytes)}` : ''}
                {' · '}{job.progress_data.files_completed ?? 0} {tr('个文件完成', 'files completed')}
                {job.status === 'running' && <>{' · '}<span aria-label={tr('预计剩余时间', 'Estimated time remaining')}>{etaLabel}</span></>}
              </small>}
              {job.status === 'running' && job.progress_data?.downloaded_bytes == null &&
                <small className="download-transfer-stats" aria-label={tr('预计剩余时间', 'Estimated time remaining')}>{etaLabel}</small>}
              {job.error && <p className="download-item-error">{job.error.message}</p>}
            </div>
            {onTrack && <button type="button" disabled={busyId !== null} onClick={() => onTrack(job)}>{tr('跟踪任务', 'Track job')}</button>}
            {onAction && (isActiveDownload(job) || retry && allowRetry) && <button disabled={busyId !== null || job.status === 'cancelling' || retry && canRetry != null && !canRetry(job)} onClick={(event) => {
              const rect = event.currentTarget.getBoundingClientRect();
              void onAction!(job, retry, { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
            }} type="button">{retry ? tr('重试', 'Retry') : tr('取消', 'Cancel')}</button>}
            {onRemove && !isActiveDownload(job) && <button disabled={busyId !== null} aria-label={tr('删除下载记录', 'Delete download record')}
              title={tr('仅删除任务记录，保留下载文件', 'Remove task history; keep downloaded files')}
              onClick={() => void onRemove!(job)} type="button"><Icon name="trash" size={16} /></button>}
          </article>
        );
      })}
    </section>
  );
}
