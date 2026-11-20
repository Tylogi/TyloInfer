import type { LoadContextDefaults } from './loadContextDefaults';
import type { ModelArtifact, RuntimeStatus } from '../../shared/api/types';
import { modelsApi } from '../../shared/api/resources/models';
import { ModelLoadContextView } from './ModelLoadContextView';

export function ModelLoadContext({ value, onChange, disabled, models = [], deviceIds = [] }: {
  value: LoadContextDefaults; onChange(value: LoadContextDefaults): void; disabled?: boolean;
  models?: ModelArtifact[]; runtime?: RuntimeStatus | null; deviceIds?: string[];
}) {
  const model = models[0];
  return model ? <ModelLoadContextView key={model.id} value={value} onChange={onChange} disabled={disabled}
    model={model} device={deviceIds[0]} readPlan={modelsApi.modelLoadPlan} /> : null;
}
