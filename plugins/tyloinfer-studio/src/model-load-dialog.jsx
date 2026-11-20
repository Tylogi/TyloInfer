import React, { useState } from 'react';
import { ModelLoadContextView } from '../../../TyloInferStudio/src/features/models/ModelLoadContextView.tsx';
import { ModelLoadPolicy } from '../../../TyloInferStudio/src/features/models/ModelLoadPolicy.tsx';
import { Icon, SettingRow } from '../../../TyloInferStudio/src/app/display.tsx';
import { Dialog } from './shared-primitives.jsx';
import { PageTabs } from './page-navigation.jsx';

import { nativeLoadRequest } from './model-load-options.js';
import { companionsAvailable, validCompanionPaths } from './companion-options.js';
export function ModelLoadDialog({ model, mode, options, onOptions, runtime, api, disabled, t, tr, onClose, onLoad, triggerRef, companionIds = [], artifacts = [] }) {
  const [tab, setTab] = useState('context');
  const [ready, setReady] = useState(false);
  const patch = value => onOptions({ ...options, ...value });
  const pathCompanions = validCompanionPaths(companionIds);
  const validCompanions = (pathCompanions || companionsAvailable(artifacts, model.id, companionIds)) && (!model.gguf_companion_required || companionIds.length > 0);
  const configuring = mode === 'configure';
  const catalog = { runtime, artifacts: [model], settingsModel: model.name, loadContext: options.context, setLoadContext: context => patch({ context }), loadPinned: options.pin, setLoadPinned: pin => patch({ pin }), loadIdleTtl: options.ttl, setLoadIdleTtl: ttl => patch({ ttl }), loadDeviceIds: options.devices, setLoadDeviceIds: devices => patch({ devices }) };
  return <Dialog open onOpenChange={open => { if (!open && !disabled) onClose(); }}
    title={configuring ? model.name : tr('加载模型', 'Load model')} description={configuring ? undefined : model.name}
    closeLabel={tr('关闭', 'Close')} className="model-settings-dialog model-load-dialog" returnFocusRef={triggerRef}>
    {mode === 'configure' && <PageTabs items={[["context", tr('上下文与 KV', 'Context and KV')], ["residency", tr('驻留与设备', 'Residency and device')]]} value={tab} onChange={setTab} label={tr('模型设置分类', 'Model settings sections')} />}
    <form className="model-settings-body tylo-model-confirm" aria-label={t('confirmLoad')} onSubmit={event => { event.preventDefault(); if (!disabled) { if (mode === 'configure') onClose(); else if (ready && validCompanions) void onLoad(nativeLoadRequest(model, options), companionIds); } }}>
      {companionIds.length > 0 && <SettingRow title={tr('已选 Engram 配套资产', 'Selected Engram companion')} detail={tr('加载时由核心核实文件与兼容性。', 'The core verifies files and compatibility on load.')} trailing={<strong>{(pathCompanions ? companionIds : companionIds.map(id => artifacts.find(item => item.id === id)?.name || tr('资产已不可用', 'Artifact unavailable'))).join(' · ')}</strong>} />}
      {mode !== 'configure' && !validCompanions && <p role="alert">{tr('请选择已登记的配套资产，或填写服务端完整路径。', 'Select a registered companion or enter its full server path.')}</p>}
      <fieldset disabled={disabled}>
        <section role="tabpanel" aria-label={tab === 'context' ? tr('上下文与 KV', 'Context and KV') : tr('驻留与设备', 'Residency and device')}>
          {tab === 'context' ? <ModelLoadContextView value={options.context} onChange={context => patch({ context })} disabled={disabled}
            model={model} device={options.devices[0]} readPlan={api.loadPlan} onReadyChange={setReady} /> : <ModelLoadPolicy catalog={catalog} />}
        </section>
        <div className="model-settings-actions model-load-actions">
          <button type="button" onClick={onClose}>{t('dismiss')}</button>
          <button type="submit" className="primary model-load-confirm-action" disabled={!configuring && (!ready || !validCompanions)}>
            {!configuring && <Icon name="play" size={14} />}
            {configuring ? tr('完成', 'Done') : tr('加载模型', 'Load model')}
          </button>
        </div>
      </fieldset>
    </form>
  </Dialog>;
}
