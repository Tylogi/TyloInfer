import { AnalysisWorkspaceView } from './AnalysisWorkspaceView';
import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router';
import { errorMessage } from '../../app/formatters';
import { useRuntime } from '../../app/RuntimeProvider';
import { modelsApi } from '../../shared/api/resources/models';
import type { CheckpointAnalysis, ModelArtifact } from '../../shared/api/types';
import { useSettings } from '../settings/SettingsProvider';

export function AnalysisPage() {
  const { tr } = useSettings();
  const { selectedModel, connectionRevision } = useRuntime();
  const [search, setSearch] = useSearchParams();
  const [models, setModels] = useState<ModelArtifact[]>([]);
  const [selected, setSelected] = useState(search.get('checkpoint') ?? '');
  const [analysis, setAnalysis] = useState<CheckpointAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [listing, setListing] = useState(true);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let current = true;
    setListing(true); setError(''); setModels([]); setAnalysis(null);
    void modelsApi.modelArtifacts(revision > 0).then(items => {
      if (!current) return;
      setModels(items);
      setSelected(previous => items.some(item => item.id === previous) ? previous
        : items.find(item => item.name === selectedModel)?.id ?? items.find(item => item.complete)?.id ?? items[0]?.id ?? '');
    }).catch(cause => { if (current) setError(errorMessage(cause)); })
      .finally(() => { if (current) { setListing(false); setLoading(false); } });
    return () => { current = false; };
  }, [connectionRevision, revision]);
  useEffect(() => {
    if (listing || !selected) return;
    const controller = new AbortController();
    setLoading(true); setAnalysis(null); setError('');
    const checkpoint = models.find(item => item.id === selected);
    if (checkpoint && !checkpoint.complete) {
      setError(tr('分片不完整，补全文件后才能分析模型。', 'Model shards are incomplete. Restore the missing files before analysis.'));
      setLoading(false);
      return;
    }
    void modelsApi.checkpointAnalysis(selected, controller.signal).then(result => {
      if (!controller.signal.aborted) setAnalysis(result);
    }).catch(cause => { if (!controller.signal.aborted) setError(errorMessage(cause)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [selected, listing, connectionRevision, revision]);
  return <AnalysisWorkspaceView models={models} selected={selected} analysis={analysis} loading={loading} listing={listing} error={error} onSelect={id => { setSelected(id); setSearch({ checkpoint: id }, { replace: true }); }} onRefresh={() => setRevision(value => value + 1)} tr={tr} />;
}
