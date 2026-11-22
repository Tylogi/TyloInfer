import type { JobResource, RuntimeInstance, RuntimeStatus } from '../../shared/api/types';

export function resolveRuntimeHero(runtime: RuntimeStatus | null | undefined, instances: RuntimeInstance[], jobs: JobResource[]) {
  const runtimeModelName = runtime?.model || 'Empty';
  const activeLoadJob = jobs.find(
    (job) =>
      job.kind === 'model.load' && ['queued', 'running', 'cancelling'].includes(job.status),
  );
  const latestLifecycleJob = jobs.find(
    (job) => job.kind === 'model.load' || job.kind === 'model.unload',
  );
  const currentInstance = instances.find((instance) => instance.id === runtime?.instance_id);
  const loadingInstance = [...instances]
    .reverse()
    .find((instance) => instance.state === 'loading');
  const readyInstance =
    currentInstance?.state === 'ready' || currentInstance?.state === 'busy'
      ? currentInstance
      : [...instances]
          .reverse()
          .find((instance) => instance.state === 'ready' || instance.state === 'busy');
  const failedInstance =
    currentInstance?.state === 'failed'
      ? currentInstance
      : [...instances].reverse().find((instance) => instance.state === 'failed');
  const jobModel =
    typeof activeLoadJob?.payload.model === 'string' ? activeLoadJob.payload.model : '';
  const runtimeState = String(runtime?.runtime_state || '').toLowerCase();

  if (runtime?.reloading || runtimeState === 'loading' || loadingInstance || activeLoadJob) {
    return {
      name: loadingInstance?.model || jobModel || runtimeModelName,
      state: 'loading' as const,
    };
  }
  if (runtimeState === 'failed' || currentInstance?.state === 'failed') {
    return {
      name: currentInstance?.model || runtimeModelName,
      state: 'failed' as const,
    };
  }
  if (runtime?.model || readyInstance) {
    return {
      name: runtime?.model || readyInstance?.model || runtimeModelName,
      state: 'ready' as const,
    };
  }
  if (failedInstance) {
    return {
      name: failedInstance.model || runtimeModelName,
      state: 'failed' as const,
    };
  }
  if (latestLifecycleJob?.kind === 'model.load' && ['failed', 'interrupted'].includes(latestLifecycleJob.status)) {
    const failedJobModel =
      typeof latestLifecycleJob.payload.model === 'string'
        ? latestLifecycleJob.payload.model
        : '';
    return {
      name: failedJobModel || runtimeModelName,
      state: 'failed' as const,
      interrupted: latestLifecycleJob.status === 'interrupted',
    };
  }
  return { name: runtimeModelName, state: 'idle' as const };
}
