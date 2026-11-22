import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ModelSettingsDialogView } from '../../../studio/web/src/features/models/ModelSettingsDialogView.tsx';
import { DFlash2Controls } from '../../../studio/web/src/features/models/DFlash2Controls.tsx';
import { ModelContextSettingsView } from '../../../studio/web/src/features/runtime/ModelContextSettingsView.tsx';
import { ConfirmationDialog } from './shared-primitives.tsx';
import { terminalStates } from './model-api.ts';
import { bytes, CoreError, Facts } from './page-kit.tsx';

function useProfiles(names, api, inventory) {
  const key = JSON.stringify(names.map(name => [name, inventory.models.find(item => item.name === name)?.id]));
  const [state, setState] = useState({ key: '', cacheProfiles: {}, cacheErrors: {} });
  useEffect(() => {
    const controller = new AbortController(); const entries = JSON.parse(key);
    setState({ key, cacheProfiles: {}, cacheErrors: {} });
    void Promise.all(entries.map(async ([name, id]) => {
      try {
        if (!id) throw new Error('Model cache metadata is unavailable');
        const value = await api.cacheProfile(id, undefined, controller.signal);
        if (!controller.signal.aborted) setState(current => ({ ...current, cacheProfiles: { ...current.cacheProfiles, [name]: value } }));
      } catch (error) {
        if (!controller.signal.aborted) setState(current => ({ ...current, cacheErrors: { ...current.cacheErrors, [name]: error.message } }));
      }
    }));
    return () => controller.abort();
  }, [key, api]);
  return state.key === key ? state : { cacheProfiles: {}, cacheErrors: {} };
}

export function LoadedModelSettings({ model, inventory, runtime, api, mutation, refresh, refreshKey, tr, onClose, triggerRef, disabled, draftDownloadAvailable, onDownloadDraft, onLoadDraft }) {
  const [confirmation, setConfirmation] = useState(null);
  const [notice, setNotice] = useState('');
  const pending = useRef(null);
  const mounted = useRef(false);
  const controller = useRef(null);
  const latest = useRef({ inventory, mutation }); latest.current = { inventory, mutation };
  const getSignal = useCallback(() => { if (!controller.current || controller.current.signal.aborted) controller.current = new AbortController(); return controller.current.signal; }, []);
  const connectionScope = useCallback(() => { const signal = getSignal(); return () => mounted.current && !signal.aborted; }, [getSignal]);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; controller.current?.abort(); pending.current?.resolve(null); pending.current = null; }; }, []);
  useEffect(() => { pending.current?.resolve(null); pending.current = null; setConfirmation(null); }, [refreshKey]);
  const requestConfirmation = useCallback((kind, request) => new Promise(resolve => {
    if (!mounted.current || pending.current) { resolve(null); return; }
    const target = { kind, request, resolve }; pending.current = target; setConfirmation(target);
  }), []);
  function cancel() { const target = pending.current; pending.current = null; setConfirmation(null); target?.resolve(null); }
  async function confirm() {
    const target = pending.current;
    if (!target || confirmation !== target || latest.current.mutation.busy || latest.current.mutation.unknown) return;
    if (!latest.current.inventory.instances.some(item => item.id === target.request.instance_id && item.model === model.name && ['ready', 'busy'].includes(item.state))) { cancel(); return; }
    pending.current = null; setConfirmation(null);
    const current = latest.current.mutation;
    const result = target.kind === 'context'
      ? await current.run(signal => api.configureContext(target.request, signal), 'context')
      : await current.run(signal => api.configureInstance(target.request.instance_id, target.request.policy, signal));
    target.resolve(result);
  }
  const services = useMemo(() => ({
    runtimeApi: {
      yarnContextInfo: id => api.yarn(id, getSignal()),
      contextPolicy: () => api.read('context_policy', {}, getSignal()),
      configureContextPolicy: request => latest.current.mutation.run(signal => api.configure('context_policy', request, signal)),
      reloadGlobalContextCap: () => Promise.reject(new Error('Global context reload is not exposed by this client')),
    },
    connectionScope,
    toast: { success: message => { if (mounted.current) setNotice(message); }, warning: message => { if (mounted.current) setNotice(message); }, error: message => { if (mounted.current) setNotice(message); } },
    useModelCacheProfiles: names => useProfiles(names, api, latest.current.inventory),
  }), [api, connectionScope, getSignal]);
  const jobs = mutation.operation?.kind === 'runtime.context.configure' ? [mutation.operation] : [];
  const runtimeState = {
    instances: inventory.instances, runtime, ready: true, connectionRevision: 0, reloadingInstances: {}, addJob() {},
    reloadModelContext: (instanceId, contextSize, yarnEnabled, qsaKvOffload, kvQuantization, kvMemory) => requestConfirmation('context', {
      instance_id: instanceId, context_size: contextSize,
      ...(yarnEnabled === undefined ? {} : { yarn_enabled: yarnEnabled }), ...(qsaKvOffload === undefined ? {} : { qsa_kv_offload: qsaKvOffload }),
      ...(kvQuantization === undefined ? {} : { kv_quantization: kvQuantization }), ...(kvMemory === undefined ? {} : { kv_memory: kvMemory }),
    }),
  };
  const blocked = disabled || mutation.busy || mutation.unknown || Boolean(mutation.operation && !terminalStates.has(mutation.operation.status));
  const instance = inventory.instances.find(item => item.model === model.name && item.state !== 'failed');
  const catalog = { artifacts: inventory.models, instances: inventory.instances, settingsModel: model.name, connectionRevision: 0, runtime,
    modelSettingsTriggerRef: triggerRef, setSettingsModel: value => { if (!value) onClose(); }, busy: blocked };
  return <>
    <ModelSettingsDialogView catalog={catalog} refreshRuntime={async () => refresh()} connectionScope={connectionScope}
      configureInstance={(instanceId, policy) => requestConfirmation('residency', { instance_id: instanceId, policy })} disabled={Boolean(blocked)}
      dflash2={<DFlash2Controls model={model} artifacts={inventory.models} instance={instance} backend={runtime?.backend}
        disabled={Boolean(blocked)} loadReady={Boolean(instance)} downloadAvailable={draftDownloadAvailable} serviceEnabled={runtime?.mtp_service_enabled}
        onDownload={onDownloadDraft} onLoad={draftId => onLoadDraft(draftId, instance.id)} tr={tr}
        onToggle={instance ? async enabled => {
          const result = await mutation.run(signal => api.configureInstance(instance.id, { dflash2_enabled: enabled }, signal));
          if (result) await refresh();
          return result;
        } : undefined} />}
      externalError={<>{mutation.ownNotice && mutation.error && <CoreError error={mutation.error} />}{mutation.unknown && <p role="alert">{tr('结果未知，不会自动重发。关闭此窗口后刷新并核实实例状态，再决定下一步。', 'Outcome unknown. The request will not be replayed. Close this dialog, refresh and verify the instance before another operation.')}</p>}</>}
      renderLoadContext={() => null} renderContext={name => <><ModelContextSettingsView scope="model" modelName={name} runtimeState={runtimeState} jobs={jobs} services={services} writeBlocked={Boolean(blocked)} />{notice && <p role="status">{notice}</p>}</>} />
    <ConfirmationDialog open={Boolean(confirmation)} tr={tr}
      title={confirmation?.kind === 'context' ? tr('应用上下文设置', 'Apply context settings') : tr('应用驻留策略', 'Apply residency policy')}
      description={confirmation?.kind === 'context' ? tr('将重载运行实例并清空活动 KV。提交后无法取消。', 'This reloads the instance and clears its active KV. The reload cannot be cancelled after submission.') : tr('立即更新模型的固定和空闲卸载策略，无需重载。', 'Apply pinning and idle-unload settings immediately. The model stays loaded.')}
      confirmLabel={confirmation?.kind === 'context' ? tr('重载并应用', 'Reload and apply') : tr('应用', 'Apply')}
      onCancel={cancel} onConfirm={confirm} disabled={mutation.busy || mutation.unknown} danger={confirmation?.kind === 'context'}>
      {confirmation?.kind === 'residency' ? <Facts entries={[
        [tr('模型', 'Model'), model.name],
        [tr('固定到内存', 'Pin in memory'), confirmation.request.policy.pinned ? tr('是', 'Yes') : tr('否', 'No')],
        [tr('空闲卸载', 'Unload when idle'), confirmation.request.policy.idle_ttl_seconds == null ? tr('永不卸载', 'Never unload') : `${confirmation.request.policy.idle_ttl_seconds} ${tr('秒', 's')}`],
      ]} /> : <Facts entries={[
          [tr('模型', 'Model'), model.name],
          [tr('最大上下文', 'Maximum context'), confirmation?.request.context_size == null ? tr('自动', 'Automatic') : `${confirmation.request.context_size} tokens`],
          ...(confirmation?.request.yarn_enabled == null ? [] : [['YaRN', confirmation.request.yarn_enabled ? tr('启用', 'Enabled') : tr('关闭', 'Disabled')]]),
          ...(confirmation?.request.qsa_kv_offload ? [[tr('流式 KV', 'Streaming KV'), confirmation.request.qsa_kv_offload.enabled ? tr('启用', 'Enabled') : tr('关闭', 'Disabled')], [tr('热层预算', 'Hot-tier budget'), bytes(confirmation.request.qsa_kv_offload.budget_bytes)], [tr('RAM 预算', 'RAM budget'), confirmation.request.qsa_kv_offload.ram_budget_bytes == null ? tr('自动', 'Automatic') : bytes(confirmation.request.qsa_kv_offload.ram_budget_bytes)]] : []),
          ...(confirmation?.request.kv_quantization ? [[tr('KV 量化', 'KV quantization'), confirmation.request.kv_quantization.enabled ? `${confirmation.request.kv_quantization.algorithm} · ${confirmation.request.kv_quantization.bits} bit` : tr('关闭', 'Disabled')]] : []),
          ...(confirmation?.request.kv_memory ? [[tr('KV 存储', 'KV placement'), confirmation.request.kv_memory.placement], [tr('KV 分配', 'KV allocation'), confirmation.request.kv_memory.allocation], ...(confirmation.request.kv_memory.qsa_mode ? [['QSA', confirmation.request.kv_memory.qsa_mode]] : [])] : []),
        ]} />}
    </ConfirmationDialog>
  </>;
}
