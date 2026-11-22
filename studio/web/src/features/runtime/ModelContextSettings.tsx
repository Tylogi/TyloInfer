import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useJobStore } from '../../stores/jobStore';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { toast } from '../../stores/toastStore';
import { useModelCacheProfiles } from '../models/useModelCacheProfiles';
import { ModelContextSettingsView } from './ModelContextSettingsView';
export function ModelContextSettings({ scope = 'all', modelName }: { scope?: 'all' | 'global' | 'model'; modelName?: string }) {
  const runtimeState = useRuntime();
  const connectionScope = useConnectionScope();
  const jobs = useJobStore(state => state.jobs);
  return <ModelContextSettingsView scope={scope} modelName={modelName} runtimeState={runtimeState} jobs={jobs}
    services={{ runtimeApi, connectionScope, toast, useModelCacheProfiles }} />;
}
