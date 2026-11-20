import { ModelLoadMemoryView } from './ModelLoadMemoryView';
import { useEffect, useState } from 'react';
import type { ModelArtifact, ModelCacheProfile, RuntimeStatus } from '../../shared/api/types';
import { modelsApi } from '../../shared/api/resources/models';
import { availableLoadMemory, estimateLoadMemory } from './loadMemoryEstimate';
import type { LoadContextDefaults } from './loadContextDefaults';

export function ModelLoadMemoryEstimate({ models, runtime, deviceIds, value, valid, disabled, onSparseChange }: {
  models: ModelArtifact[]; runtime?: RuntimeStatus | null; deviceIds: string[];
  value: LoadContextDefaults; valid: boolean; disabled: boolean;
  onSparseChange?: (sparse: boolean | null) => void;
}) {
  const [selectedId, setSelectedId] = useState('');
  const [data, setData] = useState<{ key: string; profile: ModelCacheProfile | null } | null>(null);
  const choices = models.filter(model => model.complete && model.loadable);
  const selected = choices.length === 1 ? choices[0] : choices.find(model => model.id === selectedId);
  const key = selected ? `${selected.id}:${selected.modified_at}:${selected.total_bytes}` : '';
  useEffect(() => {
    onSparseChange?.(null);
    if (!selected) { setSelectedId(''); return; }
    const controller = new AbortController();
    let active = true;
    void modelsApi.modelCacheProfile(selected.id, controller.signal, true)
      .then(profile => { if (active) { setData({ key, profile }); onSparseChange?.(profile ? profile.components.some(component => component.group === 'QSA') : null); } })
      .catch(() => { if (active) setData({ key, profile: null }); });
    return () => { active = false; controller.abort(); };
  }, [selected?.id, key]);
  const available = availableLoadMemory(runtime, deviceIds);
  const loading = !!selected && data?.key !== key;
  const estimate = selected && data?.key === key && data.profile && valid
    ? estimateLoadMemory(selected, data.profile, value) : null;
  return <ModelLoadMemoryView selected={selected} choices={choices} showChoice={models.length !== 1} available={available} loading={loading}
    estimate={estimate} valid={valid} disabled={disabled} onSelect={setSelectedId} />;
}
