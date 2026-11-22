import { DownloadQueueView } from './DownloadQueueView';
import { useState } from 'react';
import type { JobResource } from '../../shared/api/types';
import { jobsApi } from '../../shared/api/resources/jobs';
import { errorMessage } from '../../app/formatters';
import { useJobStore } from '../../stores/jobStore';
import { toast } from '../../stores/toastStore';
import { useSettings } from '../settings/SettingsProvider';
import type { DownloadOrigin } from './ModelBrowser';
import { useConnectionScope } from '../../app/useConnectionScope';

export { isModelDownload, isActiveDownload, downloadProgress, downloadEtaSeconds } from './DownloadQueueView';
export function DownloadQueue({ jobs, onJobCreated }: {
  jobs: JobResource[];
  onJobCreated(job: JobResource, origin: DownloadOrigin): void;
}) {
  const { tr } = useSettings();
  const connectionScope = useConnectionScope();
  const addJob = useJobStore((state) => state.addJob);
  const [busyId, setBusyId] = useState<string | null>(null);


  async function act(job: JobResource, retry: boolean, origin: DownloadOrigin) {
    const current = connectionScope();
    if (busyId) return;
    setBusyId(job.id);
    try {
      const next = retry ? await jobsApi.retryJob(job.id) : await jobsApi.cancelJob(job.id);
      if (!current()) return;
      if (retry && next.id !== job.id) onJobCreated(next, origin); else addJob(next);
    } catch (cause) {
      if (current()) toast.error(errorMessage(cause, tr));
    } finally {
      if (current()) setBusyId(null);
    }
  }

  async function remove(job: JobResource) {
    const current = connectionScope();
    if (busyId) return;
    setBusyId(job.id);
    try {
      await jobsApi.deleteJob(job.id);
      if (!current()) return;
      useJobStore.getState().setJobs(useJobStore.getState().jobs.filter((item) => item.id !== job.id));
    } catch (cause) { if (current()) toast.error(errorMessage(cause, tr)); }
    finally { if (current()) setBusyId(null); }
  }

  return <DownloadQueueView jobs={jobs} busyId={busyId} onAction={(job, retry, origin) => { void act(job, retry, origin); }} onRemove={job => { void remove(job); }} tr={tr} />;
}
