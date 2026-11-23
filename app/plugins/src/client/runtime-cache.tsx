import React, { useEffect, useState } from 'react';
import { PrefixCacheOverview } from '@shared/features/runtime/PrefixCacheOverview.tsx';
import { ConfirmationDialog } from './shared-primitives.tsx';
import { bytes, Facts } from './page-kit.tsx';
import { isOperationId, terminalStates } from './model-api.ts';

export function RuntimeCache({ runtime, inventory, api, mutation, refreshKey, tr }) {
  const [confirmation, setConfirmation] = useState(null);
  const [target, setTarget] = useState('');
  useEffect(() => { setConfirmation(null); setTarget(''); }, [refreshKey, runtime?.instance_id]);
  const instance = inventory?.instances.find(item => item.id === runtime?.instance_id && item.model === runtime?.model);
  const blocked = mutation.busy || mutation.unknown || Boolean(mutation.operation && !terminalStates.has(mutation.operation.status));
  const usable = isOperationId(instance?.id) && instance?.state === 'ready' && Number(runtime?.active_requests || 0) === 0;
  const targetBytes = target.trim() === '' ? null : Math.round(Number(target) * 2 ** 30);
  const validTarget = Number.isSafeInteger(targetBytes) && targetBytes >= 0;
  function prepare(kind) {
    if (!usable || blocked) return;
    const hotBytes = runtime.prefix_cache_hot_tier === 'ram' ? runtime.prefix_cache_ram_bytes ?? runtime.prefix_cache_hot_bytes : runtime.prefix_cache_hot_bytes;
    setTarget(''); setConfirmation({ kind, instance_id: instance.id, model: instance.model, hotBytes: hotBytes ?? runtime.prefix_cache_bytes });
  }
  async function confirm() {
    const current = confirmation;
    if (!current || blocked || !usable || instance.id !== current.instance_id || instance.model !== current.model || current.kind === 'trim' && !validTarget) return;
    setConfirmation(null);
    await mutation.run(signal => current.kind === 'clear' ? api.clearCache(current.instance_id, signal) : api.trimCache(current.instance_id, targetBytes, signal));
  }
  return <>
    <PrefixCacheOverview runtime={runtime} busy={blocked || !usable} onClear={() => prepare('clear')} onTrim={instance ? () => prepare('trim') : undefined} />
    <ConfirmationDialog open={Boolean(confirmation)} tr={tr}
      title={confirmation?.kind === 'trim' ? tr('缩减热缓存', 'Trim hot cache') : tr('清除运行时缓存', 'Clear runtime cache')}
      description={confirmation?.kind === 'trim' ? tr('设置当前模型要保留的热缓存上限。', 'Set the amount of hot cache to retain for the current model.') : tr('清除当前实例的前缀缓存和服务的图像解码缓存。后续请求可能需要重算。', 'Clear this instance’s prefix cache and the service’s image decode cache. Later requests may need to recompute them.')}
      confirmLabel={confirmation?.kind === 'trim' ? tr('缩减缓存', 'Trim cache') : tr('清除缓存', 'Clear cache')}
      onCancel={() => setConfirmation(null)} onConfirm={confirm} disabled={blocked || !usable || confirmation?.kind === 'trim' && !validTarget} danger>
      <Facts entries={[[tr('模型', 'Model'), confirmation?.model], [tr('运行实例', 'Runtime instance'), confirmation?.instance_id], [tr('当前热缓存', 'Current hot cache'), bytes(confirmation?.hotBytes)]]} />
      {confirmation?.kind === 'trim' && <label>{tr('保留上限（0 表示清空）', 'Retained limit (0 to empty)')}
        <span><input aria-label={tr('保留热缓存上限 (GiB)', 'Retained hot cache limit (GiB)')} type="number" min="0" step="any" value={target} onChange={event => setTarget(event.target.value)} />GiB</span>
      </label>}
    </ConfirmationDialog>
  </>;
}
