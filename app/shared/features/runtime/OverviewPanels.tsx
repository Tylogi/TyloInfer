import { Icon, ModelMonogram, SectionLabel, TMPanel } from '@shared/ui/display';
import { ModelVendorMark } from '@shared/ui/ModelVendorMark';
import { formatNumber } from '@shared/formatters';
import type { RuntimeInstance, RuntimeStatus } from '@shared/types/api';

type Translate = (zh: string, en: string) => string;
export function OverviewModels({ names, instances, runtime, selectedModel, busy, selectModel, tr }: {
  names: string[]; instances: RuntimeInstance[]; runtime?: RuntimeStatus | null; selectedModel: string;
  busy: boolean; selectModel(name: string): void; tr: Translate;
}) {
  if (names.length <= 1) return null;
  return <>
    <SectionLabel title={tr('已加载模型', 'Loaded models')} subtitle={tr(`${names.length} 个可用于推理`, `${names.length} available for inference`)} />
    <TMPanel className="overview-models-panel"><div className="overview-model-grid">
      {names.map(name => {
        const instance = instances.find(candidate => candidate.model === name && candidate.state !== 'failed');
        const selected = name === selectedModel;
        const stateLabel = instance?.state === 'busy' ? tr('使用中', 'Busy') : tr('就绪', 'Ready');
        const details = [instance?.devices.join(' + '), instance?.context_size ? `${formatNumber(instance.context_size)} ctx` : null,
          instance ? tr(`${instance.active_sessions} 个会话`, `${instance.active_sessions} sessions`) : null].filter(Boolean).join(' · ');
        return <button aria-pressed={selected} className={`overview-model-card${selected ? ' selected' : ''}`} disabled={busy || selected} key={name} onClick={() => selectModel(name)} type="button">
          <ModelMonogram name={name} state="ready" /><span className="overview-model-copy"><strong title={name}>{name}</strong><small>{details || stateLabel}</small></span>
          <span className="model-identity-trailing"><ModelVendorMark name={name} architecture={name === runtime?.model ? runtime.model_capabilities?.architecture_family || runtime.model_type : undefined} />
            <span className="runtime-status-pill ready"><i />{selected ? tr('当前', 'Current') : stateLabel}</span></span>
        </button>;
      })}
    </div></TMPanel>
  </>;
}
export function OverviewEndpoints({ endpoints, copied, copyEndpoint, tr }: {
  endpoints: { id: string; name: string; label: string; url: string | null }[];
  copied: string | null; copyEndpoint(id: string, url: string): void; tr: Translate;
}) {
  return <TMPanel className="overview-endpoint-panel">
    <div className="overview-panel-title"><Icon name="link" size={15} /><h2>{tr('API 端点', 'API endpoints')}</h2></div>
    <div className="overview-api-list">{endpoints.map(({ id, name, label, url }) => <div className="overview-api-entry" key={id} data-protocol={id}>
      <strong title={label}>{name}</strong>{url ? <code title={url}>{url}</code> : <span>{tr('暂不可用', 'Unavailable')}</span>}
      <button aria-label={tr(`复制 ${name} 端点`, `Copy ${name} endpoint`)} className={copied === id ? 'copied' : ''} disabled={!url} onClick={() => { if (url) copyEndpoint(id, url); }} type="button"><Icon name={copied === id ? 'check' : 'copy'} size={14} /></button>
    </div>)}</div>
    <p>{tr('Jev 与 OpenAI 共用端口。客户端需显式向 Jev 地址发送 POST 请求，不能使用 chat/completions。', 'Jev shares the OpenAI port. Send POST requests to the Jev URL explicitly; chat/completions does not support Jev decisions.')}</p>
  </TMPanel>;
}
