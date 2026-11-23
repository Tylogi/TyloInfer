import { runtimeApi } from '../../shared/api/resources/runtime';
import { useConnectionScope } from '../../app/useConnectionScope';
import { toast } from '../../stores/toastStore';
import { ModelAliasMappingView } from '@shared/features/connections/ModelAliasMappingView';
const services = { modelAliases: () => runtimeApi.modelAliases(), configureModelAliases: (aliases: Record<string, string>) => runtimeApi.configureModelAliases(aliases) };
export function ModelAliasMapping({ models, selectedModel }: { models: string[]; selectedModel: string }) {
  const connectionScope = useConnectionScope();
  return <ModelAliasMappingView models={models} selectedModel={selectedModel} services={services} connectionScope={connectionScope} notify={message => toast.success(message)} />;
}
