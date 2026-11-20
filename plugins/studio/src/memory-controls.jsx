import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MemoryBudgetControlsView } from '../../../studio/web/src/features/connections/MemoryBudgetControlsView.tsx';
import { PrefixCacheControlsView } from '../../../studio/web/src/features/connections/PrefixCacheControlsView.tsx';
import { hasSeparateVram } from '../../../studio/web/src/features/runtime/memoryArchitecture.ts';
import { SectionLabel, SettingRow, TMPanel } from '../../../studio/web/src/app/display.tsx';
import { Dialog } from './shared-primitives.jsx';
import { CacheDirectoryControl } from './cache-directory.jsx';
import { bytes, CoreError, Facts } from './page-kit.jsx';
import { terminalStates } from './model-api.js';

const addJob = () => {};
export function MemoryControls({ values, inventory, api, mutation, tr, refreshKey, cache = false }) {
  const [confirmation, setConfirmation] = useState(null);
  const [notice, setNotice] = useState('');
  const pending = useRef(null);
  const controller = useRef(null);
  const latest = useRef({ mutation }); latest.current = { mutation };
  const getSignal = useCallback(() => { if (!controller.current || controller.current.signal.aborted) controller.current = new AbortController(); return controller.current.signal; }, []);
  const connectionScope = useCallback(() => { const signal = getSignal(); return () => !signal.aborted; }, [getSignal]);
  const cancel = useCallback(() => { const target = pending.current; pending.current = null; setConfirmation(null); target?.resolve(null); }, []);
  useEffect(() => { cancel(); setNotice(''); return () => { controller.current?.abort(); pending.current?.resolve(null); pending.current = null; }; }, [refreshKey, cache, cancel]);
  const services = useMemo(() => ({
    memoryPolicy: () => api.read('memory_policy', {}, getSignal()),
    getJob: id => api.serviceGet(id, getSignal()),
    configureMemoryPolicy: request => new Promise(resolve => {
      if (pending.current || getSignal().aborted) { resolve(null); return; }
      const target = { request: { ...request }, resolve }; pending.current = target; setConfirmation(target);
    }),
  }), [api, getSignal]);
  async function confirm() {
    const target = pending.current; const current = latest.current.mutation;
    if (!target || target !== confirmation || current.busy || current.unknown || current.operation && !terminalStates.has(current.operation.status)) return;
    pending.current = null; setConfirmation(null);
    target.resolve(await current.run(signal => api.configure('memory_policy', target.request, signal), 'memory'));
  }
  const runtime = values.status?.value;
  const separate = hasSeparateVram(runtime);
  const loaded = (inventory?.instances || []).filter(item => ['ready', 'busy'].includes(item.state));
  const amounts = loaded.map(item => item.memory?.resident_weight_bytes ?? item.resident_bytes);
  const total = amounts.reduce((sum, value) => sum + (value ?? 0), 0);
  const residency = total > 0 ? `${amounts.some(value => value == null) ? '≥ ' : ''}${bytes(total)}` : '—';
  const blocked = mutation.busy || mutation.unknown || Boolean(mutation.operation && !terminalStates.has(mutation.operation.status));
  const jobs = [...(values.jobs?.value?.data || []), ...(mutation.operation?.kind ? [mutation.operation] : [])];
  const common = { runtime, jobs, services, connectionScope, addJob, notify: setNotice, writeBlocked: blocked };
  const labels = { total_limit_bytes: tr('总常驻预算', 'Total resident budget'), model_limit_bytes: tr('模型驻留预算', 'Model residency budget'), prefix_limit_bytes: tr('热前缀预算', 'Hot prefix budget'), prefix_ram_limit_bytes: tr('RAM 前缀预算', 'RAM prefix budget'), prefix_disk_limit_bytes: tr('SSD 前缀预算', 'SSD prefix budget') };
  return <>
    <SectionLabel title={cache ? tr('持久化前缀缓存', 'Persistent prefix cache') : separate ? tr('显存与内存规划', 'VRAM and RAM plan') : tr('内存规划', 'Memory plan')} />
    <TMPanel className={`server-settings-panel ${cache ? 'server-prefix-panel' : 'server-memory-panel'}`}><div className="setting-list">
      {values.memory_policy?.error ? <CoreError error={values.memory_policy.error} /> : values.memory_policy?.value ? cache ? <>
        <SettingRow title={tr('前缀缓存层级', 'Prefix cache hierarchy')} detail={separate ? tr('显存热层 + 低优先级 RAM 层 + 可选 SSD 冷层。', 'VRAM hot tier + low-priority RAM tier + optional SSD cold tier.') : tr('RAM 热层 + SSD 冷层。', 'RAM hot tier + SSD cold tier.')} trailing={<span className="server-managed-value">{separate ? 'VRAM → RAM → SSD' : 'RAM → SSD'}</span>} />
        <SettingRow title={tr('前缀块大小', 'Prefix block size')} detail={tr('较大的块有利于长提示词吞吐，较小的块提供更细粒度的部分前缀复用。', 'Larger blocks favor long-prompt throughput; smaller blocks allow finer partial-prefix reuse.')} trailing={<div className="server-unit-value"><strong>{runtime?.prefix_cache_block_tokens ?? '—'}</strong><span>tokens</span></div>} />
        <PrefixCacheControlsView key={refreshKey} {...common} budget={runtime?.prefix_cache_total_disk_max_bytes ?? runtime?.prefix_cache_disk_max_bytes ?? 0} connectionRevision={refreshKey} allowDirectoryEdit={false} directoryControl={<CacheDirectoryControl api={api} mutation={mutation} tr={tr} refreshKey={refreshKey} inventory={inventory} currentDirectory={values.memory_policy.value.actual_prefix_directory} />} />
        <p className="configuration-effect">{tr('目录只能从核心提供的位置中选择；更改目录需要单独确认全部模型的重载影响。', 'Choose only locations supplied by the core. Folder changes require separate confirmation of the reload impact on all models.')}</p>
      </> : <MemoryBudgetControlsView key={refreshKey} {...common} residency={residency}>
        {separate && <SettingRow title={tr('流式 KV 层级', 'Streamed KV hierarchy')} detail={tr('显存中的冷 KV 先卸载至 RAM；RAM 层按 LRU 淘汰到 SSD。RAM 前缀缓存优先让出内存。', 'Cold VRAM KV first offloads to RAM; the RAM tier evicts to SSD by LRU. RAM prefix caches yield memory first.')} trailing={<span className="server-managed-value">VRAM → RAM → SSD</span>} />}
        <SettingRow title={tr('启动时预热专家缓存', 'Warm expert cache on launch')} detail={tr('TyloInfer 会依据架构和当前内存压力自动预热可用专家槽位。', 'TyloInfer warms available expert slots according to the architecture and current memory pressure.')} trailing={<span className="server-managed-value">{tr('自动', 'Automatic')}</span>} />
      </MemoryBudgetControlsView> : <p role="status">…</p>}
      {notice && <p role="status">{notice}</p>}
    </div></TMPanel>
    <Dialog open={Boolean(confirmation)} onOpenChange={open => { if (!open) cancel(); }} title={tr('应用内存与缓存规划', 'Apply memory and cache plan')} closeLabel={tr('关闭', 'Close')}>
      <div className="tylo-model-confirm"><p>{tr('调整所有模型共用的预算，可能回收前缀缓存或改变专家驻留。提交后自动显示原生作业进度；受理不代表完成。', 'Change budgets shared by all models. This may evict prefix caches or change expert residency. Native job progress appears automatically; acceptance is not completion.')}</p>
        <Facts entries={Object.entries(confirmation?.request || {}).map(([key, value]) => [labels[key] || key, value == null ? tr('自动', 'Automatic') : bytes(value)])} />
        <div className="tylo-model-confirm-actions"><button type="button" onClick={cancel}>{tr('取消', 'Cancel')}</button><button type="button" className="primary" disabled={blocked} onClick={() => void confirm()}>{tr('确认应用', 'Confirm changes')}</button></div>
      </div>
    </Dialog>
  </>;
}
