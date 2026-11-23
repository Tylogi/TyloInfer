import React, { useEffect, useRef, useState } from 'react';
import { PrefixCacheInventoryView, defaultCacheFilters } from '@shared/features/runtime/PrefixCacheInventoryView.tsx';
import { ConfirmationDialog } from './shared-primitives.tsx';
import { bytes, CoreError, number } from './page-kit.tsx';

export function CacheInventory({ entry, api, mutation, tr, refreshKey, refresh }) {
  const [selected, setSelected] = useState(null);
  const [filters, setFilters] = useState(defaultCacheFilters);
  const [offset, setOffset] = useState(0);
  const [revision, setRevision] = useState(0);
  const [detail, setDetail] = useState(null);
  const [text, setText] = useState(null);
  const [textBlock, setTextBlock] = useState(null);
  const [reading, setReading] = useState(false);
  const [error, setError] = useState(null);
  const [confirmation, setConfirmation] = useState(null);
  const [textConsent, setTextConsent] = useState(false);
  const request = useRef(null);
  const inventory = entry?.value || null;
  useEffect(() => { setConfirmation(null); setTextConsent(false); setText(null); setTextBlock(null); }, [selected, refreshKey]);
  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController(); request.current = controller;
    setDetail(null); setText(null); setTextBlock(null); setReading(true); setError(null);
    const timer = setTimeout(() => {
      void api.read('prefix_cache', { namespace: selected, offset, limit: 100, ...filters }, controller.signal).then(value => {
        if (!controller.signal.aborted) setDetail(value);
      }).catch(error => { if (!controller.signal.aborted) setError(error); }).finally(() => { if (!controller.signal.aborted) setReading(false); });
    }, 250);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [selected, offset, filters, api, refreshKey, revision]);
  useEffect(() => () => request.current?.abort(), []);
  async function read(block, tokenOffset = 0) {
    if (!selected || reading) return;
    if (!textConsent) { setConfirmation({ kind: 'text', namespace: selected, block, offset: tokenOffset }); return; }
    await readText(selected, block, tokenOffset);
  }
  async function readText(namespace, block, tokenOffset) {
    request.current?.abort(); const controller = new AbortController(); request.current = controller;
    setReading(true); setTextBlock(block); setText(null); setError(null);
    try { const value = await api.cacheText(namespace, block, tokenOffset, controller.signal); if (!controller.signal.aborted) setText(value); }
    catch (error) { if (!controller.signal.aborted) setError(error); }
    finally { if (!controller.signal.aborted) setReading(false); }
  }
  function inspect(namespace, nextOffset = 0) {
    request.current?.abort(); setSelected(namespace); setOffset(nextOffset); setTextConsent(false); setRevision(value => value + 1);
    if (namespace !== selected) setFilters(defaultCacheFilters);
  }
  function close() { request.current?.abort(); setSelected(null); setDetail(null); setText(null); setTextBlock(null); setReading(false); setError(null); setConfirmation(null); setTextConsent(false); }
  function clear(namespace) {
    if (mutation.busy || mutation.unknown || !inventory?.can_clear) return;
    const group = namespace ? inventory.data.find(item => item.id === namespace) : null;
    if (namespace && !group) return;
    setConfirmation({ kind: 'purge', target: namespace ? { scope: 'namespace', namespace } : { scope: 'all' }, name: group?.model_name || tr('全部前缀缓存', 'All prefix caches'), blocks: group?.blocks ?? inventory.total_blocks, bytes: group?.bytes ?? inventory.total_bytes });
  }
  async function confirm() {
    const current = confirmation; if (!current) return; setConfirmation(null);
    if (current.kind === 'text') { setTextConsent(true); await readText(current.namespace, current.block, current.offset); return; }
    if (!inventory?.can_clear || mutation.busy || mutation.unknown) return;
    const result = await mutation.run(signal => api.purgeCache(current.target, signal));
    if (result) close();
  }
  return <>
    {error && <CoreError error={error} />}
    <PrefixCacheInventoryView inventory={inventory} detail={detail} selected={selected} text={text} textBlock={textBlock}
      error={entry?.error?.message || ''} busy={mutation.busy || mutation.unknown} reading={reading} filters={filters}
      refresh={refresh} inspect={inspect} read={read} clear={clear} updateFilters={next => { request.current?.abort(); setFilters(next); setOffset(0); setTextConsent(false); }} onClose={close} />
    <ConfirmationDialog open={Boolean(confirmation)} tr={tr}
      title={confirmation?.kind === 'text' ? tr('查看缓存文本', 'View cached text') : tr('清理前缀缓存', 'Clear prefix caches')}
      description={confirmation?.kind === 'text' ? tr('缓存文本可能包含对话内容。仅在本页显示所选缓存块。', 'Cached text may contain conversation content. This opens the selected block on this page.') : tr('永久删除所选的持久前缀块及文本。后续请求可能需要重算。', 'Permanently delete the selected prefix blocks and their cached text. Later requests may need to recompute them.')}
      confirmLabel={confirmation?.kind === 'text' ? tr('查看文本', 'View text') : tr('清理缓存', 'Clear caches')}
      onCancel={() => setConfirmation(null)} onConfirm={confirm} disabled={mutation.busy || mutation.unknown} danger={confirmation?.kind === 'purge'}>
      {confirmation?.kind === 'purge' && <><div className="tylo-action-summary">
        <strong>{confirmation.name}</strong><span>{number(confirmation.blocks)} {tr('个块', 'blocks')} · {bytes(confirmation.bytes)}</span>
      </div>{confirmation.target.scope === 'all' && <p>{tr('同时清理各就绪实例的运行时缓存。', 'This also clears runtime caches of ready instances.')}</p>}</>}
    </ConfirmationDialog>
  </>;
}
