import React, { useEffect, useState } from 'react';
import { PrefixCacheOverview } from '../../../studio/web/src/features/runtime/PrefixCacheOverview.tsx';
import { Dialog } from './shared-primitives.jsx';
import { bytes, Facts } from './page-kit.jsx';
import { isOperationId, terminalStates } from './model-api.js';

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
    setTarget(''); setConfirmation({ kind, instance_id: instance.id, model: instance.model, hotBytes: runtime.prefix_cache_hot_bytes ?? runtime.prefix_cache_bytes });
  }
  async function confirm() {
    const current = confirmation;
    if (!current || blocked || !usable || instance.id !== current.instance_id || instance.model !== current.model || current.kind === 'trim' && !validTarget) return;
    setConfirmation(null);
    await mutation.run(signal => current.kind === 'clear' ? api.clearCache(current.instance_id, signal) : api.trimCache(current.instance_id, targetBytes, signal));
  }
  return <>
    <PrefixCacheOverview runtime={runtime} busy={blocked || !usable} onClear={() => prepare('clear')} onTrim={instance ? () => prepare('trim') : undefined} />
    <Dialog open={Boolean(confirmation)} onOpenChange={open => { if (!open) setConfirmation(null); }} title={confirmation?.kind === 'trim' ? tr('缩减热缓存', 'Trim hot cache') : tr('清除运行时缓存', 'Clear runtime cache')} closeLabel={tr('关闭', 'Close')}>
      <div className="tylo-model-confirm"><strong>{confirmation?.model}</strong><Facts entries={[[tr('运行实例', 'Runtime instance'), confirmation?.instance_id], [tr('当前实例热缓存', 'Current instance hot cache'), bytes(confirmation?.hotBytes)]]} />
        {confirmation?.kind === 'trim' ? <><p>{tr('指定保留的目标热缓存占用上限，不是请求释放的大小。不会改预算、删除其它实例缓存或重载模型；不支持的后端会明确拒绝。', 'Set the target upper bound of retained hot cache, not the amount to release. Budgets, other instances and model loading are unchanged. Unsupported backends reject this operation.')}</p>
          <label className="server-input-unit">{tr('保留上限', 'Retained limit')}<input aria-label={tr('保留热缓存上限 (GiB)', 'Retained hot cache limit (GiB)')} type="number" min="0" step="any" value={target} onChange={event => setTarget(event.target.value)} /><span>GiB</span></label>
          <p>{validTarget ? `${tr('目标保留上限', 'Target retained limit')}: ${bytes(targetBytes)}` : tr('请输入非负容量；0 表示明确请求不保留热缓存。', 'Enter a nonnegative capacity; 0 explicitly requests no retained hot cache.')}</p>
        </> : <p>{tr('清理这个实例的运行时前缀缓存，并释放全服务的图像解码缓存。不会删除模型文件或聊天记录；后续请求可能重新处理前缀或图像。', 'Clear this instance’s runtime prefix cache and release the image decode cache across the entire service. Model files and chat history are kept; later requests may reprocess prefixes or images.')}</p>}
        <div className="tylo-model-confirm-actions"><button type="button" onClick={() => setConfirmation(null)}>{tr('取消', 'Cancel')}</button><button type="button" className="danger" disabled={blocked || !usable || confirmation?.kind === 'trim' && !validTarget} onClick={() => void confirm()}>{tr('确认清理', 'Confirm clearing')}</button></div>
      </div>
    </Dialog>
  </>;
}
