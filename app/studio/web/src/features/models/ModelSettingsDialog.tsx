import { useEffect, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { jobsApi } from '../../shared/api/resources/jobs';
import { getApiBaseUrl } from '../../shared/api/client';
import { ModelContextSettings } from '../runtime/ModelContextSettings';
import { jevEndpoint } from '../runtime/endpoint';
import { ModelLoadContext } from './ModelLoadContext';
import { ModelSettingsDialogView } from '@shared/features/models/ModelSettingsDialogView';
import { DFlash2Controls, flashNextDraft } from '@shared/features/models/DFlash2Controls';
import { useJobStore } from '../../stores/jobStore';
import { useSettings } from '../settings/SettingsProvider';
import type { useModelCatalog } from './useModelCatalog';
export function ModelSettingsDialog({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { refreshRuntime, studio } = useRuntime();
  const connectionScope = useConnectionScope();
  const artifact = catalog.artifacts.find(model => model.name === catalog.settingsModel);
  const instance = catalog.instances.find(item => item.model === artifact?.name && item.state !== 'failed');
  const { tr } = useSettings();
  const addJob = useJobStore(state => state.addJob);
  const downloadPending = useJobStore(state => state.jobs.some(job => job.kind === 'download.huggingface'
    && job.payload?.repo_id === flashNextDraft.repo_id && ['queued', 'running', 'cancelling'].includes(job.status)));
  const [downloadAvailable, setDownloadAvailable] = useState<boolean>();
  const supportsDraft = artifact?.architecture.toLowerCase().startsWith('qwen4_exp');
  useEffect(() => {
    if (!supportsDraft) return;
    let active = true;
    setDownloadAvailable(undefined);
    void jobsApi.jobKinds().then(kinds => {
      if (active) setDownloadAvailable(kinds.some(item => item.kind === 'download.huggingface'));
    }).catch(() => { if (active) setDownloadAvailable(false); });
    return () => { active = false; };
  }, [supportsDraft, catalog.connectionRevision]);
  return <ModelSettingsDialogView catalog={catalog} refreshRuntime={() => refreshRuntime(false)} connectionScope={connectionScope}
    configureInstance={runtimeApi.configureInstance} apiEndpoint={jevEndpoint(studio?.service_url || getApiBaseUrl() || window.location.origin)}
    dflash2={artifact && <DFlash2Controls key={artifact.id} model={artifact} artifacts={catalog.artifacts} instance={instance}
      backend={catalog.runtime?.backend} serviceEnabled={catalog.runtime?.mtp_service_enabled} disabled={catalog.busy}
      downloadPending={downloadPending} downloadAvailable={downloadAvailable} tr={tr}
      onDownload={async () => {
        const current = connectionScope();
        const job = await jobsApi.createJob('download.huggingface', flashNextDraft);
        if (current()) addJob(job);
        return job;
      }}
      onLoad={draftId => catalog.loadArtifact(artifact.name, undefined, { id: draftId, instanceId: instance?.id })}
      onToggle={instance ? async enabled => {
        const updated = await runtimeApi.configureInstance(instance.id, { dflash2_enabled: enabled });
        await refreshRuntime(false);
        return updated;
      } : undefined} />}
    renderContext={name => <ModelContextSettings scope="model" modelName={name} />}
    renderLoadContext={() => <ModelLoadContext value={catalog.loadContext} onChange={catalog.setLoadContext} disabled={catalog.busy || !!catalog.instances?.find(item => item.model === catalog.settingsModel && item.state !== 'failed')} models={artifact ? [artifact] : []} runtime={catalog.runtime} deviceIds={catalog.loadDeviceIds} />} />;
}
