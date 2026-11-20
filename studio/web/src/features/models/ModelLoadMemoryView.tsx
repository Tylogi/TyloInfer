import type { ModelArtifact } from '../../shared/api/types';
import { formatNumber } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import type { estimateLoadMemory } from './loadMemoryEstimate';

function gib(bytes: number | null) { return bytes == null ? '—' : `${formatNumber(bytes / 2 ** 30, 1)} GiB`; }

export function ModelLoadMemoryView({ selected, choices, showChoice, available, loading, estimate, valid, disabled, onSelect }: {
 selected?: ModelArtifact; choices: ModelArtifact[]; showChoice: boolean; available: number | null; loading: boolean;
 estimate: ReturnType<typeof estimateLoadMemory> | null; valid: boolean; disabled: boolean; onSelect(id: string): void;
}) {
 const { tr } = useSettings();
 const exceeded = estimate != null && available != null && estimate.total > available;
  return <div className="model-load-memory-estimate">
    {showChoice && <p>{tr('选择模型，比较预计占用与剩余推理预算。自动预算已考虑系统预留和可回收内存。',
      'Select a model to compare estimated usage with the remaining inference budget. The automatic budget accounts for system headroom and reclaimable memory.')}</p>}
    <div className="model-load-estimate-row">
      <strong>{tr('预计总占用（显存 + 内存）', 'Estimated total (VRAM + RAM)')}</strong>
      <div className="model-load-estimate-values">
        <span className="model-load-memory-available">{tr('当前剩余', 'Available now')} <strong>{gib(available)}</strong></span>
        <output aria-label={tr('预计总占用', 'Estimated total memory')} aria-live="polite"
          className={exceeded ? 'model-load-memory-exceeded' : undefined}
          title={estimate ? tr(`权重与预留 ${gib(estimate.weights)}，上下文缓存与递推状态 ${gib(estimate.caches)}（${estimate.context.toLocaleString()} tokens）。实际运行占用会随工作区变化。`,
            `Weights and reserve ${gib(estimate.weights)}, context caches and recurrent state ${gib(estimate.caches)} (${estimate.context.toLocaleString()} tokens). Runtime usage varies with workspaces.`) : undefined}>
          {!selected ? '—' : loading ? tr('估算中…', 'Estimating…') : !valid ? '—'
            : estimate ? gib(estimate.total) : tr('暂无法估算', 'Estimate unavailable')}
        </output>
        {showChoice && <select aria-label={tr('估算模型', 'Model to estimate')} value={selected?.id ?? ''} disabled={disabled}
          onChange={event => onSelect(event.target.value)}>
          <option value="">{tr('无', 'None')}</option>
          {choices.map(model => <option key={model.id} value={model.id}>{model.name}</option>)}
        </select>}
      </div>
    </div>
  </div>;
}
