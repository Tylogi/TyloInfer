import { useEffect, useState } from 'react';
import type { LoadContextDefaults } from './loadContextDefaults';
import type { ModelArtifact, RuntimeStatus } from '../../shared/api/types';
import { ModelLoadMemoryEstimate } from './ModelLoadMemoryEstimate';
import { ModelLoadContextView } from './ModelLoadContextView';
import { runtimeApi, type RuntimeContextPolicy } from '../../shared/api/resources/runtime';
import './model-load-context.css';

export function ModelLoadContext({ value, onChange, disabled = false, models = [], runtime, deviceIds = [] }: {
 value: LoadContextDefaults; onChange(value: LoadContextDefaults): void; disabled?: boolean;
 models?: ModelArtifact[]; runtime?: RuntimeStatus | null; deviceIds?: string[];
}) {
 const [policy, setPolicy] = useState<RuntimeContextPolicy | null>(null);
 const model = models.length === 1 ? models[0] : undefined;
 useEffect(() => {
   let active = true;
   if (model) void runtimeApi.contextPolicy().then(value => { if (active) setPolicy(value); }).catch(() => {});
   return () => { active = false; };
 }, [model?.id]);
 return <ModelLoadContextView value={value} onChange={onChange} disabled={disabled} model={model} policy={policy} backend={runtime?.backend}
   renderMemoryEstimate={props => <ModelLoadMemoryEstimate models={models} runtime={runtime} deviceIds={deviceIds} disabled={disabled} {...props} />} />;
}
