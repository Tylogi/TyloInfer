import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';

import { SettingRow } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import type { runtimeApi as NativeRuntimeApi } from '../../shared/api/resources/runtime';
import { Dialog } from '../../shared/ui/Dialog';
import { Switch } from '../../shared/ui/Switch';
import { useSettings } from '../settings/SettingsProvider';


import { ModelLoadPolicy } from './ModelLoadPolicy';


import type { useModelCatalog } from './useModelCatalog';

export function ModelSettingsDialogView({ catalog, refreshRuntime, connectionScope, configureInstance, renderContext, renderLoadContext, apiEndpoint, disabled = false, externalError, dflash2 }: {
  catalog: ReturnType<typeof useModelCatalog>; refreshRuntime: () => Promise<unknown>; connectionScope: () => () => boolean;
  configureInstance: (...args: Parameters<typeof NativeRuntimeApi.configureInstance>) => Promise<Awaited<ReturnType<typeof NativeRuntimeApi.configureInstance>> | null>;
  renderContext: (modelName: string) => ReactNode; renderLoadContext: () => ReactNode; apiEndpoint?: string; disabled?: boolean; externalError?: ReactNode; dflash2?: ReactNode;
}) {
  const { tr } = useSettings();
  const [section, setSection] = useState('context');
  const [pinned, setPinned] = useState(false);
  const [ttl, setTtl] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  const name = catalog.settingsModel;
  const artifact = catalog.artifacts.find(model => model.name === name);
  const defaultSection = artifact?.structured_decisions ? 'residency' : 'context';
  const instance = catalog.instances?.find(model => model.model === name && model.state !== 'failed');
  const loaded = instance?.state === 'ready' || instance?.state === 'busy';
  const wasLoaded = useRef(false);
  if (loaded) wasLoaded.current = true;
  useEffect(() => {
    setSection(defaultSection); setSaving(false); setSaved(false); setError('');
    setPinned(instance?.pinned ?? false); setTtl(instance?.idle_ttl_seconds ?? null);
  }, [name, instance?.id, catalog.connectionRevision, defaultSection]);
  if (!name) return null;

  async function applyResidency() {
    if (!instance || !loaded || saving || disabled) return;
    const current = connectionScope();
    setSaving(true); setSaved(false); setError('');
    try {
      const configured = await configureInstance(instance.id, { pinned, idle_ttl_seconds: pinned ? null : ttl });
      if (!configured || !current()) return;
      setSaved(true);
      await refreshRuntime();
    } catch (cause) { if (current()) setError(errorMessage(cause, tr)); }
    finally { if (current()) setSaving(false); }
  }

  return <Dialog open title={name} closeLabel={tr('关闭模型设置', 'Close model settings')}
    className={loaded || wasLoaded.current ? 'model-settings-dialog' : 'model-settings-dialog model-load-dialog'} returnFocusRef={catalog.modelSettingsTriggerRef}
    onOpenChange={open => { if (!open) catalog.setSettingsModel(null); }}>
    <div className="configuration-tabs" role="tablist" aria-label={tr('模型设置分类', 'Model settings sections')}>
      {[
        ['context', tr('上下文与 KV', 'Context and KV')], ['residency', tr('驻留与设备', 'Residency and device')],
      ].map(([value, label], index, sections) => <button type="button" role="tab" key={value}
        id={`model-settings-tab-${value}`} aria-controls={`model-settings-section-${value}`}
        aria-selected={section === value} tabIndex={section === value ? 0 : -1}
        onClick={() => setSection(value)} onKeyDown={event => {
          if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
          event.preventDefault();
          const target = sections[event.key === 'Home' ? 0 : event.key === 'End' ? sections.length - 1 : (index + 1) % sections.length][0];
          setSection(target); document.getElementById(`model-settings-tab-${target}`)?.focus();
        }}>{label}</button>)}
    </div>
    <div className="model-settings-body">
      {externalError}
      {dflash2}
      {artifact?.structured_decisions && <p className="configuration-effect">
        Jev API: <code style={{ overflowWrap: 'anywhere' }}>POST {apiEndpoint || '/v1/systemone'}</code><br />
        {tr('与 OpenAI 共用端口，无需另设端口。客户端需显式调用此地址，并将 model 设为当前模型名称。',
          'Shares the OpenAI port; no separate port is needed. Call this URL explicitly and set model to this model’s name.')}
      </p>}
      <section role="tabpanel" id="model-settings-section-context" aria-labelledby="model-settings-tab-context" hidden={section !== 'context'}>
        {loaded || wasLoaded.current ? <>
          <p className="configuration-effect">{tr('保存到服务端；应用上下文或 KV 更改会重载此模型并清除活动 KV。', 'Saved on the server. Applying context or KV changes reloads this model and clears its active KV cache.')}</p>
          {renderContext(name)}
        </> : artifact ? <>
          {renderLoadContext()}
        </> : <p role="status">{tr('模型已不在本地目录中。', 'This model is no longer in the local catalog.')}</p>}
      </section>
      <section role="tabpanel" id="model-settings-section-residency" aria-labelledby="model-settings-tab-residency" hidden={section !== 'residency'}>
        {loaded || wasLoaded.current ? <>
          <p className="configuration-effect">{tr('立即应用到此运行实例，无需重载。', 'Applies immediately to this running instance without reloading.')}</p>
          <SettingRow title={tr('计算设备', 'Compute device')}
            detail={tr('更换设备需先卸载模型。', 'Unload the model before changing its device.')}
            trailing={<span>{instance?.devices?.join(' · ') || '—'}</span>} />
          <SettingRow title={tr('固定到内存', 'Pin in memory')} detail={tr('跳过 LRU 与空闲卸载', 'Skip LRU and idle eviction')}
            trailing={<Switch label={tr('固定到内存', 'Pin in memory')} checked={pinned} disabled={disabled || saving || !loaded}
              onCheckedChange={value => { setPinned(value); setSaved(false); }} />} />
          <SettingRow title={tr('空闲卸载', 'Idle unload')} detail={tr('每次使用后重新计时', 'Resets after each use')}
            trailing={<select aria-label={tr('空闲卸载', 'Idle unload')} value={ttl ?? ''} disabled={disabled || saving || !loaded || pinned}
              onChange={event => { setTtl(event.target.value ? Number(event.target.value) : null); setSaved(false); }}>
              <option value="">{tr('永不', 'Never')}</option>
              {[...new Set([300, 900, 3600, ...(ttl != null ? [ttl] : [])])].sort((a, b) => a - b).map(value => <option value={value} key={value}>{value / 60} min</option>)}
            </select>} />
          <div className="model-settings-actions"><span role="status">{saved ? tr('已应用', 'Applied') : ''}</span>
            <button type="button" disabled={disabled || saving || !loaded || pinned === !!instance?.pinned && (pinned || ttl === (instance?.idle_ttl_seconds ?? null))}
              onClick={() => void applyResidency()}>{saving ? tr('应用中…', 'Applying…') : tr('应用', 'Apply')}</button></div>
          {error && <p role="alert">{error}</p>}
        </> : <>
          <p className="configuration-effect">{tr('下次加载此模型时生效。', 'Applies on this model’s next load.')}</p>
          <ModelLoadPolicy catalog={catalog} />
        </>}
      </section>
    </div>
  </Dialog>;
}
