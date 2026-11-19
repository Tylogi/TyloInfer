import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ModelSettingsDialogView } from '../../../TyloInferStudio/src/features/models/ModelSettingsDialogView.tsx';
import { ModelContextSettingsView } from '../../../TyloInferStudio/src/features/runtime/ModelContextSettingsView.tsx';
import { Dialog } from './shared-primitives.jsx';
import { terminalStates } from './model-api.js';
import { bytes, CoreError, Facts } from './page-kit.jsx';

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

export function LoadedModelSettings({ model, inventory, runtime, api, mutation, refresh, refreshKey, tr, onClose, triggerRef, disabled }) {
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
  const catalog = { artifacts: inventory.models, instances: inventory.instances, settingsModel: model.name, connectionRevision: 0, runtime,
    modelSettingsTriggerRef: triggerRef, setSettingsModel: value => { if (!value) onClose(); }, busy: blocked };
  return <>
    <ModelSettingsDialogView catalog={catalog} refreshRuntime={async () => refresh()} connectionScope={connectionScope}
      configureInstance={(instanceId, policy) => requestConfirmation('residency', { instance_id: instanceId, policy })} disabled={Boolean(blocked)}
      externalError={<>{mutation.error && <CoreError error={mutation.error} />}{mutation.unknown && <p role="alert">{tr('结果未知，不会自动重发。关闭此窗口后刷新并核实实例状态，再决定下一步。', 'Outcome unknown. The request will not be replayed. Close this dialog, refresh and verify the instance before another operation.')}</p>}</>}
      renderLoadContext={() => null} renderContext={name => <><ModelContextSettingsView scope="model" modelName={name} runtimeState={runtimeState} jobs={jobs} services={services} writeBlocked={Boolean(blocked)} />{notice && <p role="status">{notice}</p>}</>} />
    <Dialog open={Boolean(confirmation)} onOpenChange={open => { if (!open) cancel(); }} title={confirmation?.kind === 'context' ? tr('应用上下文设置', 'Apply context settings') : tr('应用驻留策略', 'Apply residency policy')} closeLabel={tr('关闭', 'Close')}>
      <div className="tylo-model-confirm"><strong>{model.name}</strong><p>{confirmation?.kind === 'context' ? tr('重载这个运行实例并清除其活动 KV。提交后核心不支持取消；将自动跟踪到终态，不会重发。', 'Reload this runtime instance and clear its active KV. The core cannot cancel after submission. Progress is tracked to completion without replaying the request.') : tr('立即更新此实例的固定和空闲卸载策略，不重载模型。', 'Update pinning and idle-unload policy for this instance without reloading.')}</p>
        {confirmation?.kind === 'residency' ? <p>{tr('固定到内存', 'Pin in memory')}: {confirmation.request.policy.pinned ? tr('是', 'Yes') : tr('否', 'No')} · {confirmation.request.policy.idle_ttl_seconds ?? tr('永不卸载', 'Never unload')}</p> : <Facts entries={[
          [tr('最大上下文', 'Maximum context'), confirmation?.request.context_size == null ? tr('自动', 'Automatic') : `${confirmation.request.context_size} tokens`],
          ...(confirmation?.request.yarn_enabled == null ? [] : [['YaRN', confirmation.request.yarn_enabled ? tr('启用', 'Enabled') : tr('关闭', 'Disabled')]]),
          ...(confirmation?.request.qsa_kv_offload ? [[tr('流式 KV', 'Streaming KV'), confirmation.request.qsa_kv_offload.enabled ? tr('启用', 'Enabled') : tr('关闭', 'Disabled')], [tr('热层预算', 'Hot-tier budget'), bytes(confirmation.request.qsa_kv_offload.budget_bytes)], [tr('RAM 预算', 'RAM budget'), confirmation.request.qsa_kv_offload.ram_budget_bytes == null ? tr('自动', 'Automatic') : bytes(confirmation.request.qsa_kv_offload.ram_budget_bytes)]] : []),
          ...(confirmation?.request.kv_quantization ? [[tr('KV 量化', 'KV quantization'), confirmation.request.kv_quantization.enabled ? `${confirmation.request.kv_quantization.algorithm} · ${confirmation.request.kv_quantization.bits} bit` : tr('关闭', 'Disabled')]] : []),
          ...(confirmation?.request.kv_memory ? [[tr('KV 存储', 'KV placement'), confirmation.request.kv_memory.placement], [tr('KV 分配', 'KV allocation'), confirmation.request.kv_memory.allocation], ...(confirmation.request.kv_memory.qsa_mode ? [['QSA', confirmation.request.kv_memory.qsa_mode]] : [])] : []),
        ]} />}
        <div className="tylo-model-confirm-actions"><button type="button" onClick={cancel}>{tr('取消', 'Cancel')}</button><button type="button" className="primary" disabled={mutation.busy || mutation.unknown} onClick={() => void confirm()}>{tr('确认应用', 'Confirm changes')}</button></div>
      </div>
    </Dialog>
  </>;
}
