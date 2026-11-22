import { RuntimeCache } from './runtime-cache.tsx';
import { useLogHistory } from './log-history.tsx';
import { OverviewModels, OverviewEndpoints } from '../../../studio/web/src/features/runtime/OverviewPanels.tsx';
import { runtimeModelNames } from '../../../studio/web/src/features/runtime/modelSelection.ts';
import { ConfirmationDialog } from './shared-primitives.tsx';
import { terminalStates } from './model-api.ts';
import React, { useEffect, useRef, useState } from 'react';
import { ReadOnlyJobDetails, ReadOnlyLogDetails } from './log-details.tsx';
import { CacheInventory } from './cache-inventory.tsx';
import { bytes, CoreError, EmptyGuidance, number } from './page-kit.tsx';
import { Icon, SectionLabel, TMPanel } from '../../../studio/web/src/app/display.tsx';
import { RuntimeHeroView } from '../../../studio/web/src/features/runtime/RuntimeHeroView.tsx';
import { resolveRuntimeHero } from '../../../studio/web/src/features/runtime/runtimeHeroState.ts';
import { MemoryHierarchy } from '../../../studio/web/src/features/runtime/MemoryHierarchy.tsx';
import { ResourceAllocationView } from '../../../studio/web/src/features/runtime/ResourceAllocationView.tsx';
import { ResourceMonitorView } from '../../../studio/web/src/features/runtime/ResourceMonitorView.tsx';
import { ServingStatisticsPanel, ServingTotalsPanel } from '../../../studio/web/src/features/runtime/ServingStatisticsPanel.tsx';
import { residentMemoryCapacity } from '../../../studio/web/src/features/runtime/memoryArchitecture.ts';

export function OverviewContent({ values, inventory, api, tr, go, refreshKey, selectedModel, selectModel, loading }) {
  const instances = values.instances?.value?.data ?? inventory?.instances ?? [];
  const status = values.status?.value;
  const runtime = status?.instance_id && values.instances?.value &&
    !instances.some(item => item.id === status.instance_id)
    ? { ...status, instance_id: null, model: null, runtime_state: 'idle', max_context: null } : status;
  const currentModel = instances.some(item => item.model === selectedModel && ['ready', 'busy'].includes(item.state)) ? selectedModel : '';
  const [scope, setScope] = useState('session');
  const [history, setHistory] = useState({});
  useEffect(() => {
    if (scope !== 'alltime') return;
    const controller = new AbortController();
    setHistory({});
    void api.read('serving', { scope, ...(currentModel ? { model: currentModel } : {}) }, controller.signal)
      .then(value => { if (!controller.signal.aborted) setHistory({ value }); }).catch(error => { if (!controller.signal.aborted) setHistory({ error }); });
    return () => controller.abort();
  }, [scope, api, refreshKey, currentModel]);
  const statistics = selectedModel && !currentModel ? undefined : scope === 'session' ? values.serving : history;
  const [connection, setConnection] = useState(null);
  const [connectionError, setConnectionError] = useState(null);
  const [copied, setCopied] = useState(null);
  const [copyError, setCopyError] = useState(null);
  const timer = useRef(null);
  useEffect(() => { const controller = new AbortController(); setConnection(null); setConnectionError(null); setCopied(null); setCopyError(null);
    if (api.connectionInfo) void api.connectionInfo(controller.signal).then(value => { if (!controller.signal.aborted) setConnection(value); }).catch(error => { if (!controller.signal.aborted) setConnectionError(error); });
    return () => { controller.abort(); clearTimeout(timer.current); };
  }, [api, refreshKey]);
  const endpoints = [
    { id: 'openai', name: 'OpenAI', label: tr('OpenAI 兼容端点', 'OpenAI-compatible endpoint'), url: connection?.openai_base_url || null },
    { id: 'anthropic', name: 'Anthropic', label: tr('Anthropic 兼容端点', 'Anthropic-compatible endpoint'), url: connection?.anthropic_base_url ? `${connection.anthropic_base_url}/messages` : null },
    { id: 'jev', name: 'Jev', label: tr('Jev 决策端点', 'Jev decision endpoint'), url: connection?.openai_base_url ? `${connection.openai_base_url}/systemone` : null },
  ];
  async function copyEndpoint(id, url) { try { await navigator.clipboard.writeText(url); setCopyError(null); setCopied(id); clearTimeout(timer.current); timer.current = setTimeout(() => setCopied(null), 1600); } catch (error) { setCopyError(error); } }
  const jobs = (values.jobs?.value?.data || []).map(job => ({ ...job, payload: { ...job.payload, model: inventory?.models.find(item => item.id === job.payload?.model)?.name || job.payload?.model } }));
  const modelHero = resolveRuntimeHero(runtime, instances, jobs);
  const heroState = modelHero.state;
  return <>
    <RuntimeHeroView runtime={runtime} modelHero={modelHero} endpoint={endpoints[0].url} modelHeroStatus={tr({ idle: '空闲', loading: '加载中', ready: '运行中', failed: modelHero.interrupted ? '加载中断' : '加载失败' }[heroState], { idle: 'Idle', loading: 'Loading', ready: 'Running', failed: modelHero.interrupted ? 'Interrupted' : 'Failed' }[heroState])} onModels={() => go('models')} tr={tr} />
    <OverviewModels names={runtimeModelNames([], instances)} instances={instances} runtime={runtime} selectedModel={currentModel} busy={loading} selectModel={selectModel} tr={tr} />
    {values.status?.error && <CoreError error={values.status.error} />}
    <ServingStatisticsPanel data={statistics?.value || null} error={statistics?.error?.message || null} scope={scope} onScopeChange={setScope} model={currentModel} />
    {runtime && <MemoryHierarchy runtime={runtime} instances={instances} memoryCapacityBytes={residentMemoryCapacity(runtime)} compact />}
    {connectionError && <CoreError error={connectionError} />}
    {copyError && <CoreError error={copyError} />}
    <div className="overview-footer-grid"><OverviewEndpoints endpoints={endpoints} copied={copied} copyEndpoint={(id, url) => void copyEndpoint(id, url)} tr={tr} /><ServingTotalsPanel data={statistics?.value || null} scope={scope} runtime={runtime || null} model={currentModel} /></div>
  </>;
}
export function HardwareContent({ entry, runtime, tr }) {
  return <>{entry?.error && <CoreError error={entry.error} />}<ResourceMonitorView resources={entry?.value || null} runtime={runtime} tr={tr} samplingNote={tr('核心真实采样 · 每 5 秒刷新', 'Live core sample · refreshed every 5 seconds')} /></>;
}
export function ResourcesContent({ values, inventory, tr, go, api, mutation, refreshKey, refresh }) {
  const runtime = values.status?.value;
  return <>
    <ResourceAllocationView runtime={runtime} instances={values.instances?.value?.data || inventory?.instances || []} refreshError={Object.values(values).map(entry => entry.error?.message).filter(Boolean).join(' · ') || null} tr={tr} />
    <HardwareContent entry={values.resources} runtime={runtime} tr={tr} />
    <RuntimeCache runtime={values.status?.error ? null : runtime} inventory={{ ...inventory, instances: values.instances?.value?.data || inventory?.instances || [] }} api={api} mutation={mutation} refreshKey={refreshKey} tr={tr} />
    <CacheInventory entry={values.prefix_cache} api={api} mutation={mutation} tr={tr} refreshKey={refreshKey} refresh={refresh} />
  </>;
}
export function LogsContent({ values, tr, api, refreshKey, inventory, logsPaused, mutation }) {
  const [tab, setTab] = useState('logs');
  const [query, setQuery] = useState('');
  const [level, setLevel] = useState('');
  const [source, setSource] = useState('');
  const [range, setRange] = useState('24');
  const [selected, setSelected] = useState('');
  const [exportOpen, setExportOpen] = useState(false);
  const [archiveTarget, setArchiveTarget] = useState(null);
  const writeBlocked = mutation.busy || mutation.unknown || Boolean(mutation.operation && !terminalStates.has(mutation.operation.status));
  const [exportError, setExportError] = useState(null);
  const history = useLogHistory(api, values, logsPaused, refreshKey);
  const lists = { logs: history.logs, metrics: history.requests, jobs: history.jobs.slice().sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at)) };
  useEffect(() => { setExportOpen(false); setArchiveTarget(null); }, [tab, query, level, source, range, refreshKey]);
  const timestamp = item => item.request?.completed_at ?? item.updated_at ?? item.created_at ?? item.captured_at;
  const sourceId = item => item.instance_id || item.payload?.instance_id || item.result?.instance_id;
  const sourceNames = new Map((inventory?.instances || []).map(item => [item.id, item.model]));
  for (const item of lists.metrics || []) if (item.instance_id && item.model) sourceNames.set(item.instance_id, item.model);
  for (const item of lists.logs) if (item.instance_id && typeof item.fields?.model === 'string' && !sourceNames.has(item.instance_id)) sourceNames.set(item.instance_id, item.fields.model);
  for (const item of lists.jobs) { const model = item.result?.model || item.payload?.model; if (typeof model === 'string') for (const id of [item.payload?.instance_id, item.result?.instance_id]) if (typeof id === 'string' && !sourceNames.has(id)) sourceNames.set(id, model); }
  const statusLabel = value => ({ active: tr('进行中', 'Active'), queued: tr('排队中', 'Queued'), running: tr('运行中', 'Running'), cancelling: tr('取消中', 'Cancelling'), succeeded: tr('已完成', 'Completed'), failed: tr('失败', 'Failed'), cancelled: tr('已取消', 'Cancelled'), interrupted: tr('已中断', 'Interrupted') })[value] || value;
  const jobLabel = value => ({ 'model.load': tr('加载模型', 'Load model'), 'model.unload': tr('卸载模型', 'Unload model'), 'download.huggingface': tr('下载模型', 'Download model'), 'download.modelscope': tr('下载模型', 'Download model'), 'runtime.memory.configure': tr('调整内存预算', 'Adjust memory budget'), 'runtime.context.configure': tr('调整模型上下文', 'Change model context') })[value] || value;
  const metric = (value, suffix = '') => Number.isFinite(value) ? `${number(value)}${suffix}` : '—';
  const time = value => { if (value == null) return '—'; const date = new Date(typeof value === 'number' ? value * 1000 : value); return Number.isFinite(date.getTime()) ? date.toLocaleString(undefined, { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }) : '—'; };
  const cutoff = range ? (logsPaused ? history.updatedAt ?? Date.now() : Date.now()) - Number(range) * 3600000 : 0;
  const sources = [...new Set([...sourceNames.keys(), ...lists.logs.map(sourceId), ...lists.jobs.map(sourceId)].filter(Boolean))];
  const filtered = lists[tab]?.filter(item => (!level || level === 'active' && item.status && !['succeeded', 'failed', 'cancelled', 'interrupted'].includes(item.status) || item.level === level || item.status === level || item.request?.finish_reason === level) && (!source || sourceId(item) === source) && (!query || JSON.stringify(item).toLowerCase().includes(query.toLowerCase())) && (!cutoff || (typeof timestamp(item) === 'number' ? timestamp(item) * 1000 : Date.parse(timestamp(item))) >= cutoff));
  const tabs = [['logs', tr('服务事件', 'Service events')], ['metrics', tr('请求记录', 'Requests')], ['jobs', tr('任务记录', 'Jobs')]];
  const selectTab = key => { setTab(key); setLevel(''); setSource(''); setSelected(''); };
  function exportRecords() {
    setExportOpen(false);
    try { const url = URL.createObjectURL(new Blob([JSON.stringify(filtered, null, 2)], { type: 'application/json' })); const anchor = document.createElement('a'); anchor.href = url; anchor.download = `tyloinfer-${tab}-${new Date().toISOString().slice(0, 10)}.json`; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); setExportError(null); }
    catch (error) { setExportError(error); }
  }
  async function archiveRecords() {
    const target = archiveTarget; if (!target || writeBlocked) return;
    if (target.job && !history.jobs.some(job => job.id === target.job.id && terminalStates.has(job.status))) { setArchiveTarget(null); return; }
    setArchiveTarget(null);
    const result = await mutation.run(async signal => {
      const response = target.job ? await api.deleteJob(target.job.id, signal) : await api.clearCompletedJobs(signal);
      if (response !== null) throw { code: 'tyloinfer/outcome-unknown', message: 'Invalid archive response' };
      return { archived: true };
    });
    if (result) setSelected('');
  }
  const errors = Object.values(history.errors);
  return <><TMPanel className="logs-console">
    <div className="logs-console-heading"><div className="logs-tabs" role="tablist" aria-label={tr('日志类别', 'Log categories')}>{tabs.map(([key, label], index) => <button type="button" key={key} id={`tylo-log-tab-${key}`} role="tab" aria-selected={tab === key} tabIndex={tab === key ? 0 : -1} aria-controls="tylo-log-records" onClick={() => selectTab(key)} onKeyDown={event => {
      const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : null;
      if (next != null) { event.preventDefault(); selectTab(tabs[next][0]); document.getElementById(`tylo-log-tab-${tabs[next][0]}`)?.focus(); }
    }}>{label}<span>{lists[key]?.length ?? '—'}</span></button>)}</div><span className={`logs-live-state${logsPaused || errors.length ? ' paused' : ''}`}><i />{errors.length ? tr('刷新失败', 'Refresh failed') : logsPaused ? tr('已暂停', 'Paused') : tr('实时更新', 'Live updates')}</span></div>
    <div className="logs-filters"><label className="logs-search"><span>{tr('搜索', 'Search')}</span><input type="search" value={query} placeholder={tr('搜索消息、模型或 ID', 'Search messages, models, or IDs')} onChange={event => setQuery(event.target.value)} /></label>
      <label><span>{tr('来源', 'Source')}</span><select value={source} onChange={event => setSource(event.target.value)}><option value="">{tr('全部模型与服务', 'All models and service')}</option>{sources.map(value => <option key={value} value={value}>{sourceNames.get(value) || `${tr('历史实例', 'Past instance')} · ${value.slice(0, 8)}`}</option>)}</select></label>
      <label><span>{tab === 'metrics' ? tr('停止原因', 'Finish reason') : tab === 'logs' ? tr('级别', 'Level') : tr('状态', 'Status')}</span><select value={level} onChange={event => setLevel(event.target.value)}><option value="">{tr('全部', 'All')}</option>{(tab === 'logs' ? ['debug', 'info', 'warning', 'error'] : tab === 'jobs' ? ['active', 'succeeded', 'failed', 'cancelled', 'interrupted'] : [...new Set((lists.metrics || []).map(item => item.request.finish_reason).filter(Boolean))]).map(value => <option key={value} value={value}>{tab === 'jobs' ? statusLabel(value) : value}</option>)}</select></label>
      <label><span>{tr('时间', 'Time')}</span><select value={range} onChange={event => setRange(event.target.value)}><option value="">{tr('全部时间', 'All time')}</option>{[1, 24, 168].map(value => <option value={value} key={value}>{tr(`最近 ${value} 小时`, `Last ${value} hours`)}</option>)}</select></label>
    </div>
    <div className="logs-list-heading"><span>{tr(`${filtered.length} / ${lists[tab].length} 条 · 最新在前`, `${filtered.length} / ${lists[tab].length} records · newest first`)}</span><div>{tab === 'jobs' && <button type="button" className="danger" disabled={writeBlocked || !history.jobs.some(job => terminalStates.has(job.status))} onClick={() => setArchiveTarget({ scope: 'all-completed' })}>{tr('清理已结束记录', 'Clear finished records')}</button>}{(query || source || level || range !== '24') && <button type="button" onClick={() => { setQuery(''); setSource(''); setLevel(''); setRange('24'); }}>{tr('重置筛选', 'Reset filters')}</button>}<button type="button" disabled={!filtered?.length} onClick={() => setExportOpen(true)}><Icon name="download" size={13} />{tr('导出', 'Export')}</button></div></div>
    {errors.map((error, index) => <CoreError key={index} error={error} />)}
    {errors.length > 0 && history.updatedAt && <p className="logs-error" role="status">{tr('保留上次成功读取的记录', 'Showing previously retrieved records')}</p>}
    {exportError && <CoreError error={exportError} />}
    <div id="tylo-log-records" className="logs-records" role="tabpanel" aria-labelledby={`tylo-log-tab-${tab}`}>
      {!history.updatedAt && !errors.length ? <p role="status">…</p> : !filtered.length ? <div className="logs-empty"><Icon name="scroll" size={28} /><strong>{tr('没有匹配的记录', 'No matching records')}</strong><span>{tr('可以调整筛选条件，或等待新的服务活动。', 'Adjust the filters or wait for new service activity.')}</span></div> : filtered.map((item, index) => {
        const id = String(item.key || item.id || item.sequence || index);
        return <div key={id} className={`log-record ${tab === 'logs' ? 'log-event-record' : tab === 'jobs' ? 'log-job-record' : 'log-request-record'} ${item.level || ''}`}>
          <button type="button" className="log-record-toggle" aria-expanded={selected === id} onClick={() => setSelected(selected === id ? '' : id)}><time>{time(timestamp(item))}</time>
            {tab === 'logs' ? <span className={`log-level ${item.level}`}>{item.level?.toUpperCase()}</span> : tab === 'jobs' ? <span className={`log-job-state ${item.status}`}>{statusLabel(item.status)}</span> : null}
            <span className="log-record-copy"><strong>{item.message || item.model || (item.kind && jobLabel(item.kind)) || tr('请求采样', 'Request sample')}</strong><small>{sourceNames.get(sourceId(item)) || item.request?.id || item.payload?.model || item.payload?.repo_id || item.fields?.source || tr('服务', 'Service')}</small></span>
            {item.request && <span className="log-request-metric"><b>{metric(item.request.decode_tps, ' tok/s')}</b><small>{metric(item.request.prompt_tokens)} → {metric(item.request.completion_tokens)}</small></span>}
            {Number.isFinite(item.progress) && <span className="log-job-progress"><progress max={1} value={Math.max(0, Math.min(1, item.progress))} />{number(item.progress * 100)}%</span>}<span className="log-record-chevron">›</span></button>
          {selected === id && (tab === 'jobs' ? <ReadOnlyJobDetails key={`${id}:${refreshKey}:${item.updated_at}`} job={item} api={api} tr={tr} busy={writeBlocked} onRemove={job => setArchiveTarget({ job })} /> : <ReadOnlyLogDetails key={`${tab}:${id}`} value={item} request={item.request} tr={tr} />)}
        </div>;
      })}
    </div><footer className="logs-footer"><span>{tr('每 4 秒刷新 · 服务事件 / 请求快照各保留最近 500 条', 'Refreshes every 4 seconds · latest 500 service events / request snapshots')}</span><span>{history.updatedAt ? `${tr('最近读取', 'Last retrieved')} ${new Date(history.updatedAt).toLocaleTimeString()}` : '—'}</span></footer>
  </TMPanel>
    <ConfirmationDialog open={exportOpen} tr={tr} title={tr('导出日志', 'Export logs')}
      description={tr('将筛选结果保存到本机。记录可能包含请求信息、服务路径或错误详情。', 'Save the filtered records locally. They may contain request information, service paths, or error details.')}
      confirmLabel={tr('保存到本机', 'Save locally')} onCancel={() => setExportOpen(false)} onConfirm={exportRecords} disabled={!filtered.length}>
      <div className="tylo-action-summary"><strong>{number(filtered.length)} {tr('条记录', filtered.length === 1 ? 'record' : 'records')}</strong></div>
    </ConfirmationDialog>
    <ConfirmationDialog open={Boolean(archiveTarget)} tr={tr} title={tr('清理任务记录', 'Archive job records')}
      description={archiveTarget?.job ? tr('归档选中的已结束任务，保留模型和下载文件。', 'Archive the selected finished job and keep its model and download files.') : tr('归档全部已结束的任务，包括筛选范围以外的记录。活动任务会继续运行。', 'Archive all finished jobs, including records outside the current filter. Active jobs keep running.')}
      confirmLabel={tr('确认归档', 'Confirm archive')} onCancel={() => setArchiveTarget(null)} onConfirm={archiveRecords} disabled={writeBlocked} danger>
      {archiveTarget?.job && <div className="tylo-action-summary"><strong>{jobLabel(archiveTarget.job.kind)}</strong><span>{archiveTarget.job.id}</span></div>}
    </ConfirmationDialog>
  </>;
}
