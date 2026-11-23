import React, { useEffect, useState } from 'react';
import { AnalysisWorkspaceView } from '@shared/features/analysis/AnalysisWorkspaceView.tsx';

export function AnalysisContent({ inventory, api, tr, refreshKey, refresh, selectedModel, analysisSelection: selected, setAnalysisSelection: setSelected, inventoryLoading, inventoryError }) {
  const models = inventory?.models || [];
  const listing = !inventory || inventoryLoading;
  const [snapshot, setSnapshot] = useState({ loading: false, value: null, error: null, identity: '' });
  useEffect(() => {
    if (listing) return;
    if (!models.some(item => item.id === selected)) setSelected(models.find(item => item.name === selectedModel)?.id || models.find(item => item.complete)?.id || models[0]?.id || '');
  }, [models, selected, listing, selectedModel, setSelected]);
  const artifact = models.find(model => model.id === selected);
  const identity = selected + ':' + refreshKey + ':' + Boolean(artifact?.complete);
  useEffect(() => {
    const controller = new AbortController();
    setSnapshot({ loading: Boolean(selected), value: null, error: null, identity });
    if (!selected || listing || !artifact) return () => controller.abort();
    if (artifact && !artifact.complete) {
      setSnapshot({ loading: false, value: null, error: new Error(tr('分片不完整，补全文件后才能分析模型。', 'Model shards are incomplete. Restore the missing files before analysis.')), identity });
      return () => controller.abort();
    }
    void api.analysis(selected, controller.signal).then(value => {
      if (!['layers', 'tensors', 'projections', 'experts', 'warnings'].every(key => Array.isArray(value[key])) || !value.graph || !value.attention_distribution) throw new Error('Invalid analysis response');
      if (!controller.signal.aborted) setSnapshot({ loading: false, value, error: null, identity });
    }).catch(error => { if (!controller.signal.aborted) setSnapshot({ loading: false, value: null, error, identity }); });
    return () => controller.abort();
  }, [api, identity, listing]);
  const current = snapshot.identity === identity ? snapshot : { loading: Boolean(selected) };
  const error = inventoryError || current.error;
  return <AnalysisWorkspaceView models={models} selected={selected} analysis={current.value || null} loading={Boolean(current.loading)} listing={listing}
    error={error?.details?.core_error?.message || error?.message || null} onSelect={setSelected} onRefresh={refresh} tr={tr} />;
}
