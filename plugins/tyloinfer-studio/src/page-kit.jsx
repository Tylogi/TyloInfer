import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { isOperationId, terminalStates } from './model-api.js';
import { useSettings } from './shared-presenters.jsx';
import { ToastItem } from '../../../TyloInferStudio/src/shared/ui/ToastItem.tsx';
import { ApiError } from '../../../TyloInferStudio/src/shared/api/ApiError.ts';
import { errorMessage } from '../../../TyloInferStudio/src/app/formatters.ts';

export const number = value => Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—';
export const bytes = value => !Number.isFinite(value) ? '—' : value >= 2 ** 30 ? `${number(value / 2 ** 30)} GiB` : value >= 2 ** 20 ? `${number(value / 2 ** 20)} MiB` : `${number(value)} B`;
export const rows = value => Array.isArray(value?.data) ? value.data : null;

function ErrorNotice({ message, floating, dismissLabel }) {
  const [dismissed, setDismissed] = useState(false);
  const dismiss = useCallback(() => setDismissed(true), []);
  if (dismissed) return null;
  const item = <ToastItem toast={{ id: 'core-error', revision: 0, type: 'error', message, duration: 5000 }}
    onDismiss={dismiss} dismissLabel={dismissLabel} />;
  return floating ? <div className="studio-toast-container">{item}</div> : item;
}

export function CoreError({ error, floating = false }) {
  const { tr } = useSettings();
  const core = error?.details?.core_error ?? error;
  const text = errorMessage(core);
  const translated = typeof core?.code === 'string' ? errorMessage(new ApiError(error?.details?.http_status ?? 0, { error: core }), tr) : text;
  const message = translated === `${core?.code}: ${core?.message}` ? text : translated;
  return <ErrorNotice key={message} message={message} floating={floating} dismissLabel={tr('关闭通知', 'Dismiss notification')} />;
}
export function Facts({ entries }) {
  return <dl className="tylo-page-facts">{entries.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value ?? '—'}</dd></div>)}</dl>;
}
export function DataTable({ columns, items, empty }) {
  const numeric = ([key, _label, format]) => format === number || format === bytes || Boolean(items?.some(item => typeof item[key] === 'number'));
  return !items ? <p role="alert">Invalid list response</p> : !items.length ? <p className="tylo-page-empty">{empty}</p> : <div className="tylo-page-table"><table><thead><tr>{columns.map(column => <th key={column[0]} className={numeric(column) ? 'tylo-numeric' : undefined}>{column[1]}</th>)}</tr></thead>
    <tbody>{items.map((item, index) => <tr key={item.id || item.sequence || index}>{columns.map(column => { const [key, _label, format] = column; return <td key={key} className={numeric(column) ? 'tylo-numeric' : undefined}>{format ? format(item[key], item) : typeof item[key] === 'object' ? JSON.stringify(item[key]) : String(item[key] ?? '—')}</td>; })}</tr>)}</tbody></table></div>;
}
export function CoreSection({ title, entry, children }) {
  return <section className="tylo-page-section"><h2>{title}</h2>{entry?.error ? <CoreError error={entry.error} /> : entry?.value !== undefined ? children(entry.value) : <p role="status">…</p>}</section>;
}
export function EmptyGuidance({ tr, go, title }) {
  return <div className="tylo-page-guidance"><strong>{title || tr('尚未加载模型', 'No model is loaded')}</strong><p>{tr('先登记本地模型，或打开下载页查看执行能力；不会自动加载或下载权重。', 'Register a local model or open Downloads to check available capabilities. No model is loaded or downloaded automatically.')}</p>
    <button type="button" onClick={() => go('models')}>{tr('管理本地模型', 'Manage local models')}</button><button type="button" onClick={() => go('downloads')}>{tr('打开下载', 'Open downloads')}</button></div>;
}
export function usePageRead(api, view, params, refreshKey) {
  const [snapshot, setSnapshot] = useState({ loading: true, values: {}, identity: '' });
  const key = JSON.stringify(params);
  const identity = view + ':' + key + ':' + refreshKey;
  useEffect(() => {
    const controller = new AbortController();
    const polling = ['overview', 'models', 'resources', 'downloads'].includes(view);
    let timer; let pending = false;
    setSnapshot({ loading: true, values: {}, identity });
    async function read() {
      if (pending || controller.signal.aborted) return;
      clearTimeout(timer);
      if (polling && document.hidden) { timer = setTimeout(read, 5000); return; }
      pending = true;
      try {
        const values = await api.page(view, JSON.parse(key), controller.signal);
        if (!controller.signal.aborted) setSnapshot(previous => {
          if (previous.identity === identity) for (const [resource, entry] of Object.entries(values)) if (entry.error && previous.values[resource]?.value !== undefined) values[resource] = { ...entry, value: previous.values[resource].value };
          return { loading: false, values, identity };
        });
      } finally { pending = false; if (polling && !controller.signal.aborted) timer = setTimeout(read, 5000); }
    }
    function visible() { if (!document.hidden) void read(); }
    if (polling) document.addEventListener('visibilitychange', visible);
    void read();
    return () => { controller.abort(); clearTimeout(timer); if (polling) document.removeEventListener('visibilitychange', visible); };
  }, [api, view, key, refreshKey]);
  return snapshot.identity === identity ? snapshot : { loading: true, values: {} };
}

const detachedPageOutcome = saved => saved?.unknown === true && saved.id === null && saved.kind === null;

export function usePageMutation(api, refresh, refreshKey, checkpoint, view) {
  const visit = useMemo(() => ({}), [view]);
  const [state, setState] = useState(() => {
    const saved = checkpoint?.load();
    const restored = saved && ['memory', 'download', 'context'].includes(saved.kind) && isOperationId(saved.id);
    return { busy: false, error: null, result: null, operation: restored ? { id: saved.id, status: 'accepted' } : null, operationKind: saved?.kind || null,
      unknown: saved?.unknown === true && !detachedPageOutcome(saved), reviewAfter: refreshKey - 1 };
  });
  const stateRef = useRef(state);
  const controller = useRef(null);
  const active = useRef(false);
  const pending = useRef(false);
  const last = useRef(state.operationKind);
  const origin = useRef(null);
  const epoch = useRef(0);
  useEffect(() => { active.current = true; controller.current = new AbortController(); return () => { active.current = false; controller.current.abort(); }; }, []);
  useEffect(() => { if (detachedPageOutcome(checkpoint?.load())) checkpoint.save(null); }, [checkpoint]);
  const update = patch => { if (active.current) {
    stateRef.current = { ...stateRef.current, ...patch, operationKind: last.current };
    const current = stateRef.current;
    const pendingOperation = current.operation && !terminalStates.has(current.operation.status);
    const unknown = current.unknown || (current.busy && !current.operation);
    checkpoint?.save(last.current !== 'registration' && (pendingOperation || unknown) ? { id: current.operation?.id, kind: last.current, unknown } : null);
    setState(current);
  } };
  const run = async (request, kind = 'sync') => {
    if (pending.current || stateRef.current.unknown || (stateRef.current.operation && !terminalStates.has(stateRef.current.operation.status))) return null;
    pending.current = true; epoch.current++; last.current = kind; origin.current = visit;
    update({ busy: true, error: null, result: null, operation: null });
    try {
      const result = await request(controller.current.signal);
      if (!active.current) return null;
      if (kind === 'memory') {
        if (!isOperationId(result.operation_id) || result.status !== 'accepted') throw { code: 'tyloinfer/outcome-unknown', message: 'Invalid operation ticket' };
        last.current = kind;
        update({ operation: { id: result.operation_id, status: 'accepted' }, result: null });
      } else if (kind === 'download' || kind === 'context') {
        const allowedKinds = kind === 'context' ? ['runtime.context.configure'] : ['download.huggingface', 'download.modelscope'];
        if (!isOperationId(result.id) || !allowedKinds.includes(result.kind) || !['queued', 'running', 'cancelling', ...terminalStates].includes(result.status)) throw { code: 'tyloinfer/outcome-unknown', message: 'Invalid native operation' };
        last.current = kind;
        update({ operation: result, result: null });
        if (terminalStates.has(result.status)) refresh();
      } else { update({ result, operation: null }); refresh(); }
      return result;
    } catch (error) {
      const unknown = error.code === 'tyloinfer/outcome-unknown';
      update({ error, unknown: unknown && kind !== 'registration', reviewAfter: refreshKey });
      if (unknown && kind === 'registration') refresh();
      return null;
    }
    finally { pending.current = false; update({ busy: false }); }
  };
  const readOperation = async () => {
    const operation = stateRef.current.operation;
    if (!operation || pending.current || !controller.current || controller.current.signal.aborted) return;
    const generation = epoch.current;
    try {
      const next = await (last.current === 'download' ? api.downloadGet : last.current === 'context' ? api.contextGet : api.serviceGet)(operation.id, controller.current.signal);
      if (!active.current || generation !== epoch.current) return;
      if (next.id !== operation.id || !['queued', 'running', 'cancelling', ...terminalStates].includes(next.status)) throw { message: 'Invalid operation state' };
      update({ operation: next, error: null });
      if (terminalStates.has(next.status)) refresh();
    } catch (error) { if (!controller.current.signal.aborted && generation === epoch.current) update({ error }); }
  };
  useEffect(() => {
    if (!state.operation || terminalStates.has(state.operation.status) || state.error || state.busy) return;
    const timer = setTimeout(() => { void readOperation(); }, 1200);
    return () => clearTimeout(timer);
  }, [state.operation, state.busy, state.error]);
  const cancel = async () => {
    const operation = stateRef.current.operation;
    if (!operation || last.current === 'context' || !['queued', 'running'].includes(operation.status) || operation.cancel_requested || pending.current || stateRef.current.unknown) return;
    pending.current = true; epoch.current++; origin.current = visit; update({ busy: true, error: null });
    try {
      const next = await (last.current === 'download' ? api.downloadCancel : api.serviceCancel)(operation.id, controller.current.signal);
      if (next.id !== operation.id) throw { code: 'tyloinfer/outcome-unknown', message: 'Invalid cancellation response' };
      update({ operation: next });
    } catch (error) { update({ error, unknown: error.code === 'tyloinfer/outcome-unknown', reviewAfter: refreshKey }); }
    finally { pending.current = false; update({ busy: false }); }
  };
  const track = async (kind, id) => {
    if (!isOperationId(id) || pending.current || (stateRef.current.operation && !terminalStates.has(stateRef.current.operation.status))) return;
    pending.current = true; epoch.current++; origin.current = visit; update({ busy: true, error: null });
    try {
      const next = await (kind === 'download' ? api.downloadGet : api.serviceGet)(id, controller.current.signal);
      if (next.id !== id || !['queued', 'running', 'cancelling', ...terminalStates].includes(next.status)) throw { message: 'Invalid operation state' };
      last.current = kind; update({ operation: next });
      return next;
    } catch (error) { update({ error }); }
    finally { pending.current = false; update({ busy: false }); }
  };
  return { ...state, ownNotice: origin.current === visit, run, readOperation, cancel, track, clear: () => { if (!pending.current) update({ result: null, error: null, unknown: false }); } };
}
export function PageMutationStatus({ mutation, tr, refreshKey, reviewReady }) {
  const error = mutation.error ?? mutation.operation?.error;
  return <div className="tylo-page-operation">
    {mutation.busy && <p role="status">{tr('正在提交一次请求…', 'Submitting once…')}</p>}
    {mutation.ownNotice && error && <CoreError floating error={
      mutation.operationKind === 'registration' && error.code === 'tyloinfer/outcome-unknown'
        ? tr('导入未返回确认。请在刷新后的模型列表中查看结果。', 'Import was not confirmed. Check the refreshed model list for the result.')
        : error} />}
    {mutation.unknown && <div role="alert"><p>{tr('结果未知，不会自动重发。请刷新并人工核实后再操作。', 'Outcome unknown. No automatic retry. Refresh and verify manually before another operation.')}</p>
      <button type="button" disabled={refreshKey <= mutation.reviewAfter || mutation.busy || !reviewReady} onClick={mutation.clear}>{tr('已刷新并核实状态，允许新操作', 'I refreshed and verified the state; allow a new operation')}</button></div>}
    {mutation.result && <><p role="status">{mutation.result.failed_blocks > 0 ? tr(`部分完成：已删除 ${number(mutation.result.removed_blocks)} 个块，${number(mutation.result.failed_blocks)} 个块未能删除。`, `Partially completed: removed ${number(mutation.result.removed_blocks)} blocks; ${number(mutation.result.failed_blocks)} could not be removed.`) : tr('服务端已确认更新', 'Update confirmed by the server')}</p>{Number.isFinite(mutation.result.released_bytes) && <p>{tr('释放容量', 'Released')}: {bytes(mutation.result.released_bytes)}</p>}{Number.isFinite(mutation.result.image_decode_cache_released_bytes) && <p>{tr('全服务图像解码缓存释放', 'Service-wide image decode cache released')}: {bytes(mutation.result.image_decode_cache_released_bytes)}</p>}</>}
    {mutation.operation && <section><h2>{mutation.operationKind === 'download' ? tr('下载进度', 'Download progress') : mutation.operationKind === 'context' ? tr('应用模型上下文设置', 'Applying model context settings') : tr('调整内存预算', 'Updating memory budget')}</h2><p role="status">{mutation.operation.status === 'accepted' ? tr('已受理，尚未完成', 'Accepted; not completed') : mutation.operation.status}</p>
      {Number.isFinite(mutation.operation.progress) && <progress max={1} value={mutation.operation.progress} />}
      {!terminalStates.has(mutation.operation.status) && <>{mutation.error && <button type="button" disabled={mutation.busy} onClick={() => void mutation.readOperation()}>{tr('重试读取进度', 'Retry loading progress')}</button>}
        {mutation.operationKind !== 'context' && <button type="button" disabled={mutation.busy || mutation.unknown || mutation.operation.cancel_requested || !['queued', 'running'].includes(mutation.operation.status)} onClick={() => void mutation.cancel()}>{tr('取消', 'Cancel')}</button>}</>}
    </section>}
  </div>;
}
