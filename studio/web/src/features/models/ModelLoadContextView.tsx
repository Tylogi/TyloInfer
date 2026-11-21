import { useEffect, useId, useState } from 'react';
import { useSettings } from '../settings/SettingsProvider';
import type { ModelLoadPlan, ModelLoadPlanQuery } from '../../shared/api/resources/models';
import type { ModelArtifact } from '../../shared/api/types';
import type { LoadContextDefaults } from './loadContextDefaults';
import './model-load-context.css';

const tiers = [
  { reserve: 5, zh: '激进', en: 'Aggressive' },
  { reserve: 10, zh: '普通', en: 'Balanced' },
  { reserve: 20, zh: '安全', en: 'Safe' },
] as const;

type LoadPlanReader = (id: string, query: ModelLoadPlanQuery, signal: AbortSignal) => Promise<ModelLoadPlan>;

function missingPlanEndpoint(error: unknown): boolean {
  if (typeof error !== 'object' || error === null) return false;
  const response = error as { status?: number; code?: string; details?: { http_status?: number; core_error?: unknown } };
  return response.status === 404 && response.code !== 'model_artifact_not_found'
    || response.details?.http_status === 404 && response.details.core_error == null;
}

export function ModelLoadContextView({ value, onChange, disabled = false, model, device, readPlan, onReadyChange }: {
  value: LoadContextDefaults;
  onChange(value: LoadContextDefaults): void;
  disabled?: boolean;
  model: ModelArtifact;
  device?: string;
  readPlan: LoadPlanReader;
  onReadyChange?(ready: boolean): void;
}) {
  const { tr } = useSettings();
  const id = useId();
  const [draft, setDraft] = useState<string | null>(null);
  const [result, setResult] = useState<{ key: string; plan: ModelLoadPlan | null; endpointMissing?: boolean } | null>(null);
  const [attempt, setAttempt] = useState(0);
  const { context_size, reserve_percent, precision, vram_preallocated, jev_weights } = value;
  const streamed = jev_weights !== 'vram';
  const valid = draft == null || draft === '' || Number.isSafeInteger(Number(draft)) && Number(draft) >= 512 && Number(draft) <= 2147483647;
  const key = JSON.stringify([model.id, device, context_size, reserve_percent, precision, vram_preallocated, streamed, attempt]);
  const plan = result?.key === key ? result.plan : null;
  const loading = result?.key !== key;
  const ready = valid && !!plan?.usable;
  const patch = (next: Partial<LoadContextDefaults>) => onChange({ ...value, ...next });
  const automatic = context_size == null;
  const context = draft ?? (automatic ? plan?.context_size ?? '' : context_size);
  const label = tr('上下文长度', 'Context length');
  const precisionLabel = (bits: number) => bits === 16 ? tr('原生', 'Native') : bits === 8 ? 'INT8' : tr(`${bits} 位`, `${bits}-bit`);

  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(() => {
      void readPlan(model.id, {
        reserve: reserve_percent, preallocate: vram_preallocated, streamed_jev: streamed,
        ...(context_size == null ? {} : { context_size }), ...(precision == null ? {} : { precision }),
        ...(device ? { device } : {}),
      }, controller.signal).then(plan => {
        if (!controller.signal.aborted) setResult({ key, plan });
      }).catch(error => {
        if (!controller.signal.aborted) setResult({ key, plan: null, endpointMissing: missingPlanEndpoint(error) });
      });
    }, 140);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [key, model.id, readPlan, vram_preallocated, streamed, reserve_percent, context_size, precision, device]);

  useEffect(() => { onReadyChange?.(ready); }, [ready, onReadyChange]);

  const issue = !valid ? tr('请输入至少 512 的整数。', 'Enter a whole number of at least 512.')
    : loading ? '' : !plan
      ? result?.endpointMissing
        ? tr('当前核心版本缺少加载方案接口。', 'The connected core version has no load-plan endpoint.')
        : tr('无法读取加载方案。请重试。', 'Could not read the load plan. Retry to check again.')
      : plan.reason === 'memory_unavailable'
        ? tr('核心缺少可用内存或模型 KV 容量数据。', 'The core could not measure available memory or this model’s KV size.')
      : plan.reason === 'precision_unsupported'
        ? tr('请选择此模型支持的 KV 精度。', 'Select a KV precision supported by this model.')
        : plan.reason === 'model_limit'
          ? tr('超出模型支持的上下文长度。', 'The context exceeds the model limit.')
          : !plan.usable ? tr('所选设置超出可用容量。', 'These settings exceed the available memory.') : '';

  return <section className="model-load-context" aria-label={tr('加载选项', 'Load options')}>
    <fieldset className="model-load-tiers">
      <legend>{tr('内存余量', 'Memory reserve')}</legend>
      {tiers.map(tier => <label className="model-load-tier" key={tier.reserve}>
        <input type="radio" name={id} value={tier.reserve} checked={reserve_percent === tier.reserve} disabled={disabled}
          onChange={() => patch({ reserve_percent: tier.reserve })} />
        <span>{tr(tier.zh, tier.en)}</span><strong>{tier.reserve}%</strong>
      </label>)}
    </fieldset>
    <div className="model-load-inputs">
      <div className="model-load-field">
        <label htmlFor={id + '-context'}>{label}</label>
        <div className="model-load-context-input">
          <input id={id + '-context'} type="number" min={512} max={2147483647} step={1} value={context}
            aria-invalid={!valid || !!plan?.reason} disabled={disabled}
            onChange={event => {
              const raw = event.target.value;
              setDraft(raw);
              const number = Number(raw);
              if (raw === '' || Number.isSafeInteger(number) && number >= 512 && number <= 2147483647)
                patch({ context_size: raw === '' ? null : number });
            }} onBlur={() => setDraft(null)} />
          {automatic ? <span>{tr('自动', 'Auto')}</span>
            : <button type="button" disabled={disabled} onClick={() => { setDraft(null); patch({ context_size: null }); }}>{tr('恢复自动', 'Reset')}</button>}
        </div>
        <small aria-live="polite">{loading ? tr('计算中…', 'Calculating…') : plan?.max_context
          ? tr(`上限 ${plan.max_context.toLocaleString()} tokens`, `Up to ${plan.max_context.toLocaleString()} tokens`) : '\u00a0'}</small>
      </div>
      <div className="model-load-field">
        <label htmlFor={id + '-precision'}>{tr('KV 精度', 'KV precision')}</label>
        <select id={id + '-precision'} value={precision ?? ''} disabled={disabled}
          onChange={event => patch({ precision: event.target.value === '' ? null : Number(event.target.value) })}>
          <option value="">{tr('自动', 'Auto')}{plan ? ' · ' + precisionLabel(plan.precision) : ''}</option>
          {[...new Set([...(result?.plan?.precisions ?? []), ...(precision == null ? [] : [precision])])].filter(bits => bits < 16)
            .map(bits => <option key={bits} value={bits}>{precisionLabel(bits)}</option>)}
        </select>
      </div>
    </div>
    {!model.structured_decisions && <div className="model-load-allocation">
      <label htmlFor={id + '-allocation'}>{tr('KV 分配', 'KV allocation')}</label>
      <select id={id + '-allocation'} value={vram_preallocated ? 'preallocate' : 'dynamic'} disabled={disabled}
        onChange={event => patch({ vram_preallocated: event.target.value === 'preallocate' })}>
        <option value="preallocate">{tr('加载时预分配', 'Preallocate on load')}</option>
        <option value="dynamic">{tr('按需分配', 'Allocate on demand')}</option>
      </select>
    </div>}
    {issue && <div className="model-load-status">
      <p className="model-load-issue" role="alert">{issue}</p>
      {valid && (!plan || plan.reason === 'memory_unavailable') && <button type="button" disabled={disabled}
        onClick={() => setAttempt(value => value + 1)}>{tr('重试', 'Retry')}</button>}
    </div>}
  </section>;
}
