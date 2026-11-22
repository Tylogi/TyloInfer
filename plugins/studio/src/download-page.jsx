import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ModelBrowserView } from '../../../studio/web/src/features/models/ModelBrowserView.tsx';
import { DownloadQueueView } from '../../../studio/web/src/features/models/DownloadQueueView.tsx';
import { useDownloadReuseView } from '../../../studio/web/src/features/models/useDownloadReuseView.ts';
import { parseHubReference } from '../../../studio/web/src/features/models/hubReference.ts';
import { Icon } from '../../../studio/web/src/app/display.tsx';
import { bytes, CoreError, Facts, rows } from './page-kit.jsx';
import { ConfirmationDialog } from './shared-primitives.jsx';
import { DownloadDirectory } from './download-directory.jsx';

export function DownloadsContent({ values, api, mutation, tr, refreshKey }) {
  const [tab, setTab] = useState('official');
  const [error, setError] = useState('');
  const [confirmation, setConfirmation] = useState(null);
  const pending = useRef(null);
  const queueRequest = useRef(null);
  const localReadFailed = ['download_directory', 'jobs', 'job_kinds'].some(resource => Boolean(values[resource]?.error));
  const downloadMutation = { ...mutation, busy: mutation.busy || localReadFailed };
  const directory = values.download_directory?.value?.actual_directory;
  const kinds = rows(values.job_kinds?.value) || [];
  const queueRows = rows(values.jobs?.value);
  const jobs = queueRows ? [...new Map([...queueRows, ...(mutation.operation ? [mutation.operation] : [])].filter(job => ['download.huggingface', 'download.modelscope'].includes(job.kind)).map(job => [job.id, job])).values()] : null;
  const mutationRef = useRef(downloadMutation); mutationRef.current = downloadMutation;
  const context = useRef({ directory, kinds }); context.current = { directory, kinds };
  useEffect(() => () => { queueRequest.current?.abort(); pending.current?.resolve?.(null); pending.current = null; }, []);
  useEffect(() => { queueRequest.current?.abort(); pending.current?.resolve?.(null); pending.current = null; setConfirmation(null); }, [refreshKey, directory]);
  const createJob = useCallback(async (kind, payload) => {
    const state = mutationRef.current;
    if (pending.current || state.busy || state.unknown || !context.current.directory || !context.current.kinds.some(item => item.kind === kind)) return null;
    return new Promise(resolve => {
      const provider = kind === 'download.huggingface' ? 'huggingface' : kind === 'download.modelscope' ? 'modelscope' : null;
      if (!provider) { resolve(null); return; }
      const target = { type: 'create', provider, request: { ...payload, download_directory: context.current.directory }, resolve };
      pending.current = target; setConfirmation(target);
    });
  }, []);
  const services = useMemo(() => ({
    modelsApi: {
      officialHubModels: (refresh, signal) => api.official({ refresh }, signal),
      localOfficialHubModels: signal => api.officialLocal(signal),
      hubModelInfo: (provider, repoId, revision, signal) => api.info({ provider, repo_id: repoId, ...(revision ? { revision } : {}) }, signal),
      browseHubModels: (provider, query, options) => api.search({ provider, query, formats: options.formats, sort: options.sort, direction: options.direction, ...(options.cursor ? { cursor: options.cursor } : {}), limit: 40 }, options.signal),
      resolveHubModel: (reference, provider, signal) => {
        const parsed = parseHubReference(reference, provider);
        if (!parsed) return Promise.reject(new Error(tr('仅支持 Hugging Face / ModelScope 仓库引用。', 'Only Hugging Face / ModelScope repository references are supported.')));
        return api.info({ provider: parsed.provider, repo_id: parsed.repoId, ...(parsed.revision ? { revision: parsed.revision } : {}) }, signal);
      },
      hubModelReadme: (model, signal) => api.readme({ provider: model.provider, repo_id: model.repo_id, revision: model.revision }, signal),
      probeDownloadSources: (sources, signal) => api.probeSources(sources, signal),
    },
    createJob,
    openExternal: async value => { const url = new URL(value); if (!['https:', 'http:'].includes(url.protocol)) throw new Error('Invalid repository link'); window.open(url.href, '_blank', 'noopener,noreferrer'); },
    useDownloadReuse: (payload, key) => useDownloadReuseView(payload, key, api.reuse, false),
  }), [api, createJob, tr]);
  function cancel() { const current = pending.current; pending.current = null; setConfirmation(null); current?.resolve?.(null); }
  async function prepareQueueAction(job, type) {
    if (downloadMutation.busy || mutation.unknown || pending.current) return;
    queueRequest.current?.abort(); const controller = new AbortController(); queueRequest.current = controller;
    try {
      const value = await api.downloadGet(job.id, controller.signal);
      if (controller.signal.aborted) return;
      if (value.id !== job.id || !['download.huggingface', 'download.modelscope'].includes(value.kind)) throw new Error('Invalid download record');
      if (type === 'retry' && (!['failed', 'cancelled', 'interrupted'].includes(value.status) || !context.current.kinds.some(item => item.kind === value.kind))) throw new Error(tr('此任务当前不能重试，或下载执行器未启用。', 'This job cannot be retried now, or its download executor is disabled.'));
      if (type === 'delete' && !['succeeded', 'failed', 'cancelled', 'interrupted'].includes(value.status)) throw new Error(tr('仅可移除已结束的下载记录。', 'Only finished download records can be removed.'));
      const target = { type, job: value, request: value.payload || {} }; pending.current = target; setConfirmation(target);
    } catch (error) { if (!controller.signal.aborted) setError(error.message); }
  }
  async function submit() {
    const target = pending.current;
    if (!target || target !== confirmation || downloadMutation.busy || mutation.unknown) return;
    pending.current = null; setConfirmation(null);
    if (target.type === 'delete') {
      await mutation.run(async signal => { await api.downloadDelete(target.job.id, signal); return { archived: true }; });
    } else {
      const created = await mutation.run(signal => target.type === 'retry' ? api.downloadRetry(target.job.id, signal) : api.download(target.provider, target.request, signal), 'download');
      target.resolve?.(created);
    }
  }
  const queue = <>{values.jobs?.error && <><CoreError error={values.jobs.error} />{jobs && <p role="status">{tr('显示上次读取的下载队列；暂不可执行任务操作。', 'Showing the previous download queue; task actions are unavailable until reads recover.')}</p>}</>}{jobs ? <DownloadQueueView jobs={jobs} tr={tr} canRetry={job => kinds.some(item => item.kind === job.kind)} busyId={downloadMutation.busy || mutation.unknown ? 'pending' : null} onRemove={job => void prepareQueueAction(job, 'delete')} onAction={async (job, retry) => {
    if (downloadMutation.busy || mutation.unknown) return;
    if (retry) return prepareQueueAction(job, 'retry');
    if (mutation.operation?.id === job.id) return mutation.cancel();
    const tracked = await mutation.track('download', job.id); if (tracked?.id === job.id) return mutation.cancel();
  }} /> : !values.jobs?.error && <p role="status">…</p>}</>;
  return <div className="model-download-page">
    {values.job_kinds?.error && <CoreError error={values.job_kinds.error} />}
    {error && <p className="logs-error" role="alert">{error}</p>}
    {values.job_kinds?.value && !kinds.some(item => ['download.huggingface', 'download.modelscope'].includes(item.kind)) && <p className="configuration-effect" role="status">{tr('当前服务未启用下载执行器。可显式浏览模型和队列，但不能提交下载。', 'No download executor is enabled. Explicit model browsing and the queue remain available, but downloads cannot be submitted.')}</p>}
    <ModelBrowserView jobKinds={localReadFailed ? [] : kinds} onError={setError} onJobCreated={() => setTab('downloads')} tab={tab} onTabChange={setTab} downloadQueue={queue}
      downloadDirectory={directory} downloadDirectoryControl={<DownloadDirectory entry={values.download_directory} api={api} mutation={downloadMutation} tr={tr} refreshKey={refreshKey} />}
      tr={tr} services={services} initialLocalStateReady={['download_directory', 'jobs', 'job_kinds'].every(resource => Boolean(values[resource]))} autoLoadCatalog={false} autoSearch={false} allowSourceProbe allowRemoteImages={false} manualMetadataActions actionRevision={refreshKey} />
    <ConfirmationDialog open={Boolean(confirmation)} tr={tr}
      title={confirmation?.type === 'delete' ? tr('移除下载记录', 'Remove download record') : confirmation?.type === 'retry' ? tr('重试下载', 'Retry download') : tr('确认下载', 'Confirm download')}
      description={confirmation?.type === 'delete' ? tr('移除已结束的任务记录，已下载的文件会保留。', 'Remove this finished job record. Downloaded files are kept.') : tr('下载所选权重和配套文件，完成后可在模型页加载。', 'Download the selected weights and companion files. Load the model from the Models page when ready.')}
      confirmLabel={confirmation?.type === 'delete' ? tr('移除记录', 'Remove record') : confirmation?.type === 'retry' ? tr('重试下载', 'Retry download') : tr('开始下载', 'Start download')}
      onCancel={cancel} onConfirm={submit} disabled={downloadMutation.busy || mutation.unknown} danger={confirmation?.type === 'delete'}>
      <Facts entries={[[tr('仓库', 'Repository'), confirmation?.request.repo_id],
        ...(confirmation?.type === 'delete' ? [] : [[tr('下载目录', 'Download folder'), confirmation?.request.download_directory], [tr('预计大小', 'Estimated size'), bytes(confirmation?.request.expected_bytes)]])]} />
      {confirmation?.type !== 'delete' && confirmation?.request.include?.length > 0 && <ul aria-label={tr('下载文件', 'Files to download')}>{confirmation.request.include.map(pattern => <li key={pattern}><code>{pattern}</code></li>)}</ul>}
    </ConfirmationDialog>
    <button className="download-circle" type="button" aria-label={tr('转到下载队列', 'Go to download queue')} onClick={() => setTab('downloads')}><svg className="download-circle-ring" viewBox="0 0 56 56" aria-hidden="true"><circle cx="28" cy="28" r="24" /></svg><Icon name="download" size={22} />{jobs?.some(job => ['queued', 'running', 'cancelling'].includes(job.status)) && <span className="download-circle-count">{jobs.filter(job => ['queued', 'running', 'cancelling'].includes(job.status)).length}</span>}</button>
  </div>;
}
