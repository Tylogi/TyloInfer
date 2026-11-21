import { Icon, TMPanel, ModelMonogram } from '../../app/display';
import { formatNumber, formatDuration } from '../../app/formatters';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { ModelLoadProgress } from '../models/ModelLoadProgress';
import type { RuntimeStatus } from '../../shared/api/types';

export function RuntimeHeroView({ runtime, modelHero, modelHeroStatus, endpoint, onModels, tr }: {
  runtime?: RuntimeStatus | null; modelHero: { name: string; state: 'idle' | 'loading' | 'ready' | 'failed' };
  modelHeroStatus: string; endpoint?: string | null; onModels(localCheckpoints?: boolean): void; tr(zh: string, en: string): string;
}) {
  return (
    <TMPanel className="runtime-hero">
      <ModelMonogram name={modelHero.name} state={modelHero.state} />
      <div className="runtime-hero-copy">
        <div>
          <h2>{modelHero.name}</h2>
          <span className={`runtime-status-pill ${modelHero.state}`}>
            <i />
            {modelHeroStatus}
          </span>
        </div>
        <p className="runtime-endpoint">{endpoint}</p>
        <small>
          {runtime?.model
            ? `${runtime?.model_type || 'TQ'} · ${formatNumber(runtime?.max_context)} ${tr('上下文', 'context')} · ${formatDuration(runtime?.uptime_seconds)}`
            : tr('加载本地模型后即可开始推理。', 'Load a local model to begin inference.')}
        </small>
        {modelHero.state === 'loading' && <ModelLoadProgress model={modelHero.name} />}
      </div>
      <div className="runtime-hero-actions">
        <ModelVendorMark name={modelHero.name} size={34}
          architecture={modelHero.name === runtime?.model ? runtime.model_capabilities?.architecture_family || runtime.model_type : undefined} />
        <button className={modelHero.state === 'ready' ? undefined : 'primary'} onClick={() => onModels(modelHero.state === 'idle')} type="button">
          <Icon name="folder" size={15} />
          {modelHero.state === 'idle' ? tr('加载', 'Load') : tr('模型', 'Models')}
        </button>
      </div>
    </TMPanel>
  );
}
