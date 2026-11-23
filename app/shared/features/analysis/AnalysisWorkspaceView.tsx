import { AnalysisPanels } from '@shared/features/analysis/AnalysisPanels';
import { Icon, EmptyPanel } from '@shared/ui/display';
import { ModelVendorMark } from '@shared/ui/ModelVendorMark';
import { CompactSelect } from '@shared/ui/CompactSelect';
import type { CheckpointAnalysis, ModelArtifact } from '@shared/types/api';
export function AnalysisWorkspaceView({ models, selected, analysis, loading, listing, error, onSelect, onRefresh, tr }: {
  models: ModelArtifact[]; selected: string; analysis?: CheckpointAnalysis | null; loading: boolean; listing: boolean; error?: string | null;
  onSelect(id: string): void; onRefresh(): void; tr(zh: string, en: string): string;
}) {
  const selectedArtifact = models.find(item => item.id === selected);
  return <div className="analysis-page"><section className="analysis-checkpoint-bar">
    <label><span>{tr('模型', 'Model')}</span><CompactSelect aria-label={tr('选择模型', 'Select model')} disabled={listing || !models.length} value={selected} onChange={event => onSelect(event.target.value)}>
      {!models.length && <option value="">{listing ? tr('正在读取…', 'Reading…') : tr('没有本地模型', 'No local models')}</option>}
      {models.map(item => <option key={item.id} value={item.id}>{item.name}{!item.complete ? ` · ${tr('分片不全', 'Incomplete shards')}` : ''}</option>)}
    </CompactSelect></label>
    {selectedArtifact && <ModelVendorMark name={selectedArtifact.name} architecture={selectedArtifact.architecture} size={25} />}
    <button aria-label={tr('刷新模型', 'Refresh models')} disabled={listing} onClick={onRefresh} type="button"><Icon name="refresh" size={15} /></button>
  </section>
    {loading || listing ? <div className="analysis-loading" role="status"><Icon name="search" size={22} /><strong>{tr('正在读取模型元数据', 'Reading model metadata')}</strong></div>
      : error ? <div className="analysis-error" role="alert"><Icon name="info" size={18} /><div><strong>{tr('暂时无法分析', 'Analysis unavailable')}</strong><p>{error}</p></div><button onClick={onRefresh} type="button">{tr('重试', 'Retry')}</button></div>
        : !analysis ? <EmptyPanel icon="search" title={tr('选择一个模型', 'Choose a model')} message={tr('已注册的本地模型都可以在这里查看。', 'Inspect registered local models here.')} />
          : <AnalysisPanels key={analysis.model_id} analysis={analysis} selectedArtifact={selectedArtifact} tr={tr} />}
  </div>;
}
