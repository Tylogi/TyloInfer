import { PrefixCacheOverview } from '@shared/features/runtime/PrefixCacheOverview';
import { useRef, useState } from 'react';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useSettings } from '../settings/SettingsProvider';
import { ScreenHeader } from '@shared/ui/display';
import { errorMessage, formatNumber } from '@shared/formatters';
import { studioConfirm } from '../../studio';
import { ResourceMonitorPanel } from './ResourceMonitorPanel';
import { ResourceAllocationPanel } from './ResourceAllocationPanel';
import { PrefixCacheInventoryPanel } from './PrefixCacheInventoryPanel';
import { toast } from '../../stores/toastStore';

export function CachePage() {
  const { connectionRevision } = useRuntime();
  return <ResourcePageContent key={connectionRevision} />;
}

function ResourcePageContent() {
  const { runtime, refreshRuntime } = useRuntime();
  const connectionScope = useConnectionScope();
  const { tr } = useSettings();
  const [busy, setBusy] = useState(false);
  const clearing = useRef(false);
  async function clearRuntimeCache() {
    const current = connectionScope();
    const snapshots = Number(runtime?.prefix_cache_snapshots || 0);
    if (clearing.current || snapshots <= 0 || Number(runtime?.active_requests || 0) > 0) return;
    clearing.current = true;
    setBusy(true);
    const hotPrefixOnly = runtime?.prefix_cache_mode === 'single_device_hot_prefix';
    const confirmation = hotPrefixOnly
      ? tr(
          `清除当前设备热前缀（${formatNumber(snapshots)} 个）？此操作不会删除聊天记录。`,
          `Clear ${formatNumber(snapshots)} device-hot prefix? Chat history will be kept.`,
        )
      : tr(
          `清除当前模型的前缀缓存（${formatNumber(snapshots)} 个块）？其他模型的缓存和聊天记录不受影响。`,
          `Clear this model’s ${formatNumber(snapshots)} prefix cache blocks? Other models’ caches and chat history will be kept.`,
        );
    try {
      if (!(await studioConfirm(confirmation)) || !current()) return;
      await runtimeApi.clearRuntimeCache(runtime?.instance_id);
      if (!current()) return;
      await refreshRuntime(false);
      if (!current()) return;
      toast.success(tr('缓存已清除', 'Cache cleared successfully'));
    } catch (cause) {
      if (current()) toast.error(errorMessage(cause, tr));
    } finally {
      clearing.current = false;
      if (current()) setBusy(false);
    }
  }

  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={tr('资源', 'Resources')}
        subtitle={tr(
          '监视计算占用、权重与 KV 流量、前缀缓存。',
          'Monitor compute utilization, weight and KV traffic, and prefix caching.',
        )}
      />

      <ResourceAllocationPanel />
      <ResourceMonitorPanel />
      <PrefixCacheOverview runtime={runtime} busy={busy} onClear={() => void clearRuntimeCache()} />
      <PrefixCacheInventoryPanel />
    </section>
  );
}
