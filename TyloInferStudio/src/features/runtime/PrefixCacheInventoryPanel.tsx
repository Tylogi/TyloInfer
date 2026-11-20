import { useCallback, useEffect, useRef, useState } from 'react';
import { runtimeApi, type PrefixCacheInventory, type PrefixCacheText, type PrefixCacheFilters } from '../../shared/api/resources/runtime';
import { PrefixCacheInventoryView } from './PrefixCacheInventoryView';
import { useRuntime } from '../../app/RuntimeProvider';
import { hasSeparateVram } from './memoryArchitecture';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useSettings } from '../settings/SettingsProvider';

import { errorMessage, formatBytes, formatNumber } from '../../app/formatters';
import { studioConfirm } from '../../studio';
import { toast } from '../../stores/toastStore';

const defaultFilters: PrefixCacheFilters = { query: '', search_in: 'all', text_filter: 'all', chain_filter: 'all', sort: 'recent' };

export function PrefixCacheInventoryPanel() {
  const { connectionRevision } = useRuntime();
  return <CacheInventory key={connectionRevision} />;
}

function CacheInventory() {
  const { tr } = useSettings();
  const { refreshRuntime, runtime } = useRuntime();
  const connectionScope = useConnectionScope();
  const [inventory, setInventory] = useState<PrefixCacheInventory | null>(null);
  const [detail, setDetail] = useState<PrefixCacheInventory | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [text, setText] = useState<PrefixCacheText | null>(null);
  const [textBlock, setTextBlock] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [reading, setReading] = useState(false);
  const [filters, setFilters] = useState<PrefixCacheFilters>(defaultFilters);
  const searchTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const detailRequest = useRef<AbortController | null>(null);
  const clearing = useRef(false);
  const sequence = useRef(0);
  const inventorySequence = useRef(0);
  const refresh = useCallback(async (signal?: AbortSignal) => {
    const current = connectionScope();
    const version = ++inventorySequence.current;
    try {
      const value = await runtimeApi.prefixCacheEntries(undefined, 0, signal);
      if (signal?.aborted || !current() || version !== inventorySequence.current) return;
      setInventory(value);
      setError('');
    } catch (cause) {
      if (!signal?.aborted && current() && version === inventorySequence.current) setError(errorMessage(cause, tr));
    }
  }, [connectionScope]);
  useEffect(() => {
    const controller = new AbortController();
    let pending = false;
    async function update() {
      if (pending || document.hidden) return;
      pending = true;
      try { await refresh(controller.signal); } finally { pending = false; }
    }
    void update();
    const timer = window.setInterval(() => void update(), 5000);
    document.addEventListener('visibilitychange', update);
    return () => { controller.abort(); detailRequest.current?.abort(); clearTimeout(searchTimer.current); window.clearInterval(timer); document.removeEventListener('visibilitychange', update); sequence.current += 1; };
  }, [refresh]);
  async function inspect(namespace: string, offset = 0, nextFilters = namespace === selected ? filters : defaultFilters) {
    clearTimeout(searchTimer.current);
    detailRequest.current?.abort();
    const controller = new AbortController();
    detailRequest.current = controller;
    const current = connectionScope();
    const version = ++sequence.current;
    setSelected(namespace);
    setFilters(nextFilters);
    setDetail(null);
    setText(null);
    setTextBlock(null);
    setReading(true);
    try {
      const value = await runtimeApi.prefixCacheEntries(namespace, offset, controller.signal, nextFilters);
      if (version === sequence.current && current()) setDetail(value);
    } catch (cause) { if (version === sequence.current && current()) toast.error(errorMessage(cause, tr)); }
    finally { if (version === sequence.current && current()) setReading(false); }
  }
  function updateFilters(next: PrefixCacheFilters) {
    setFilters(next);
    clearTimeout(searchTimer.current);
    detailRequest.current?.abort();
    sequence.current += 1;
    setDetail(null); setText(null); setTextBlock(null);
    if (selected) {
      setReading(true);
      searchTimer.current = setTimeout(() => void inspect(selected, 0, next), 250);
    }
  }
  async function read(block: string, offset = 0) {
    if (!selected) return;
    const current = connectionScope();
    const version = ++sequence.current;
    setTextBlock(block);
    setText(null);
    setReading(true);
    try {
      const value = await runtimeApi.prefixCacheText(selected, block, offset);
      if (version === sequence.current && current()) setText(value);
    } catch (cause) { if (version === sequence.current && current()) toast.error(errorMessage(cause, tr)); }
    finally { if (version === sequence.current && current()) setReading(false); }
  }
  async function clear(namespace?: string) {
    const current = connectionScope();
    if (!inventory?.can_clear || clearing.current) return;
    const group = inventory.data.find((item) => item.id === namespace);
    const message = namespace
      ? tr(`清理此缓存组（${formatBytes(group?.bytes ?? 0)}）？模型文件和聊天记录不会删除。`,
          `Clear this cache group (${formatBytes(group?.bytes ?? 0)})? Model files and chat history will be kept.`)
      : hasSeparateVram(runtime) && runtime?.prefix_cache_hot_tier !== 'ram' ? tr(`清理全部前缀缓存（${formatBytes(inventory.total_bytes)}），包括未载入模型的 SSD 缓存、RAM 前缀缓存与显存热缓存？模型文件和聊天记录不会删除。`,
          `Clear all prefix caches (${formatBytes(inventory.total_bytes)}), including unloaded models’ SSD caches, RAM prefixes and VRAM hot caches? Model files and chat history will be kept.`) : tr(`清理全部前缀缓存（${formatBytes(inventory.total_bytes)}），包括未载入模型的 SSD 缓存与 RAM 热缓存？模型文件和聊天记录不会删除。`,
          `Clear all prefix caches (${formatBytes(inventory.total_bytes)}), including unloaded models’ SSD caches and RAM hot caches? Model files and chat history will be kept.`);
    clearing.current = true;
    setBusy(true);
    try {
      if (!await studioConfirm(message) || !current()) return;
      const result = await runtimeApi.purgePrefixCache(namespace);
      if (!current()) return;
      sequence.current += 1;
      clearTimeout(searchTimer.current);
      detailRequest.current?.abort();
      setSelected(null); setDetail(null); setText(null); setTextBlock(null); setReading(false);
      await Promise.all([refresh(), refreshRuntime(false)]);
      if (!current()) return;
      const summary = tr(`已清理 ${formatNumber(result.removed_blocks)} 个块，释放 ${formatBytes(result.released_bytes)}`,
        `Cleared ${formatNumber(result.removed_blocks)} blocks; released ${formatBytes(result.released_bytes)}`);
      if (result.failed_blocks) toast.error(`${summary} · ${tr('部分块未能删除', 'Some blocks could not be removed')}`);
      else toast.success(summary);
    } catch (cause) { if (current()) toast.error(errorMessage(cause, tr)); }
    finally { clearing.current = false; if (current()) setBusy(false); }
  }
  return <PrefixCacheInventoryView inventory={inventory} detail={detail} selected={selected} text={text} textBlock={textBlock}
    error={error} busy={busy} reading={reading} filters={filters} refresh={() => void refresh()}
    inspect={(namespace, offset) => void inspect(namespace, offset)} read={(block, offset) => void read(block, offset)}
    clear={namespace => void clear(namespace)} updateFilters={updateFilters}
    onClose={() => { sequence.current += 1; clearTimeout(searchTimer.current); detailRequest.current?.abort(); setSelected(null); setDetail(null); setText(null); setReading(false); }} />;
}
