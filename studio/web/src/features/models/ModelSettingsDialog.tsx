import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { getApiBaseUrl } from '../../shared/api/client';
import { ModelContextSettings } from '../runtime/ModelContextSettings';
import { jevEndpoint } from '../runtime/endpoint';
import { ModelLoadContext } from './ModelLoadContext';
import { ModelSettingsDialogView } from './ModelSettingsDialogView';
import type { useModelCatalog } from './useModelCatalog';
export function ModelSettingsDialog({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { refreshRuntime, studio } = useRuntime();
  const connectionScope = useConnectionScope();
  const artifact = catalog.artifacts.find(model => model.name === catalog.settingsModel);
  return <ModelSettingsDialogView catalog={catalog} refreshRuntime={() => refreshRuntime(false)} connectionScope={connectionScope}
    configureInstance={runtimeApi.configureInstance} apiEndpoint={jevEndpoint(studio?.service_url || getApiBaseUrl() || window.location.origin)}
    renderContext={name => <ModelContextSettings scope="model" modelName={name} />}
    renderLoadContext={() => <ModelLoadContext value={catalog.loadContext} onChange={catalog.setLoadContext} disabled={catalog.busy || !!catalog.instances?.find(item => item.model === catalog.settingsModel && item.state !== 'failed')} models={artifact ? [artifact] : []} runtime={catalog.runtime} deviceIds={catalog.loadDeviceIds} />} />;
}
