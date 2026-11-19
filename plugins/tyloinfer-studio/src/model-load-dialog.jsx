import React, { useEffect, useState } from 'react';
import { ModelLoadContextView } from '../../../TyloInferStudio/src/features/models/ModelLoadContextView.tsx';
import { ModelLoadMemoryView } from '../../../TyloInferStudio/src/features/models/ModelLoadMemoryView.tsx';
import { ModelLoadPolicy } from '../../../TyloInferStudio/src/features/models/ModelLoadPolicy.tsx';
import { availableLoadMemory, estimateLoadMemory } from '../../../TyloInferStudio/src/features/models/loadMemoryEstimate.ts';
import { SettingRow } from '../../../TyloInferStudio/src/app/display.tsx';
import { Dialog } from './shared-primitives.jsx';
import { PageTabs } from './page-navigation.jsx';
import { CoreError } from './page-kit.jsx';

import { nativeLoadRequest } from './model-load-options.js';
import { companionsAvailable } from './companion-options.js';
function MemoryEstimate({ model, runtime, profile, loading, value, valid, disabled, devices, onSparseChange }) {
  useEffect(() => { onSparseChange(profile ? profile.components.some(item => item.group === 'QSA') : null); }, [profile, onSparseChange]);
  const estimate = profile && valid ? estimateLoadMemory(model, profile, value) : null;
  return <ModelLoadMemoryView selected={model} choices={[model]} showChoice={false} available={availableLoadMemory(runtime, devices)} loading={loading} estimate={estimate} valid={valid} disabled={disabled} onSelect={() => {}} />;
}
export function ModelLoadDialog({ model, mode, options, onOptions, runtime, api, disabled, t, tr, onClose, onLoad, triggerRef, companionIds = [], artifacts = [] }) {
  const [tab, setTab] = useState('context');
  const [metadata, setMetadata] = useState({ loading: true, policy: null, profile: null, errors: [] });
  const patch = value => onOptions({ ...options, ...value });
  const validCompanions = companionsAvailable(artifacts, model.id, companionIds) && (!model.gguf_companion_required || companionIds.length > 0);
  useEffect(() => {
    const controller = new AbortController();
    setMetadata({ loading: true, policy: null, profile: null, errors: [] });
    void Promise.allSettled([api.read('context_policy', {}, controller.signal), api.cacheProfile(model.id, true, controller.signal)]).then(results => {
      if (controller.signal.aborted) return;
      setMetadata({ loading: false, policy: results[0].status === 'fulfilled' ? results[0].value : null, profile: results[1].status === 'fulfilled' ? results[1].value : null, errors: results.filter(result => result.status === 'rejected').map(result => result.reason) });
    });
    return () => controller.abort();
  }, [model.id, api]);
  const catalog = { runtime, artifacts: [model], settingsModel: model.name, loadContext: options.context, setLoadContext: context => patch({ context, contextChanged: true }), loadPinned: options.pin, setLoadPinned: pin => patch({ pin }), loadIdleTtl: options.ttl, setLoadIdleTtl: ttl => patch({ ttl }), loadDeviceIds: options.devices, setLoadDeviceIds: devices => patch({ devices }) };
  return <Dialog open onOpenChange={open => { if (!open && !disabled) onClose(); }} title={model.name} closeLabel={tr('关闭模型设置', 'Close model settings')} className="model-settings-dialog" returnFocusRef={triggerRef}>
    <PageTabs items={[["context", tr('上下文与 KV', 'Context and KV')], ["residency", tr('驻留与设备', 'Residency and device')]]} value={tab} onChange={setTab} label={tr('模型设置分类', 'Model settings sections')} />
    <form className="model-settings-body tylo-model-confirm" aria-label={t('confirmLoad')} onSubmit={event => { event.preventDefault(); if (!disabled) { if (mode === 'configure') onClose(); else if (validCompanions) void onLoad(nativeLoadRequest(model, options), companionIds); } }}>
      <p className="configuration-effect">{tr('下次加载此模型时生效；不影响其它模型。', 'Applies on this model’s next load only; other models are unchanged.')}</p>
      {metadata.errors.map((error, index) => <CoreError error={error} key={index} />)}
      {companionIds.length > 0 && <SettingRow title={tr('已选 Engram 配套资产', 'Selected Engram companion')} detail={tr('加载前由核心按工件 ID 核实文件与兼容性，不使用客户端路径。', 'Before loading, the core verifies files and compatibility by artifact ID, not a client path.')} trailing={<strong>{companionIds.map(id => artifacts.find(item => item.id === id)?.name || tr('资产已不可用', 'Artifact unavailable')).join(' · ')}</strong>} />}
      {mode !== 'configure' && !validCompanions && <p role="alert">{tr('请返回目录并选择完整的已登记 GGUF 配套资产。', 'Return to the catalog and select a complete registered GGUF companion.')}</p>}
      <fieldset disabled={disabled}>
        <section role="tabpanel" aria-label={tab === 'context' ? tr('上下文与 KV', 'Context and KV') : tr('驻留与设备', 'Residency and device')}>
          {tab === 'context' ? <><ModelLoadContextView value={options.context} onChange={context => patch({ context, contextChanged: true })} disabled={disabled} model={model} policy={metadata.policy} renderMemoryEstimate={props => <MemoryEstimate model={model} runtime={runtime} profile={metadata.profile} loading={metadata.loading} devices={options.devices} disabled={disabled} {...props} />} />
            <details className="configuration-disclosure"><summary>{tr('预填充分块', 'Prefill chunk')}</summary><SettingRow title={tr('分块 token 数', 'Chunk size in tokens')} detail={tr('留空由核心自动决定。', 'Leave blank for the core to decide automatically.')} trailing={<input aria-label={t('prefillChunk')} type="number" min={1} max={8192} step={1} placeholder="Auto" value={options.chunk} onChange={event => patch({ chunk: event.target.value })} />} /></details>
          </> : <ModelLoadPolicy catalog={catalog} />}
        </section>
        <div className="model-settings-actions"><button type="button" onClick={onClose}>{t('dismiss')}</button><button type="submit" className="primary" disabled={mode !== 'configure' && !validCompanions}>{mode === 'configure' ? tr('完成', 'Done') : t('load')}</button></div>
      </fieldset>
    </form>
  </Dialog>;
}
