import type { RuntimeStatus } from '../../shared/api/types';
import { SectionLabel, TMPanel, UsageBar, EmptyPanel, Icon } from '../../app/display';
import { formatNumber } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { hasSeparateVram } from './memoryArchitecture';

export function PrefixCacheOverview({ runtime, busy = false, onClear, onTrim }: { runtime?: RuntimeStatus | null; busy?: boolean; onClear?(): void; onTrim?(): void }) {
 const { tr } = useSettings();
 const separate = hasSeparateVram(runtime);
 const vramPrefix = separate && runtime?.prefix_cache_hot_tier !== 'ram';
  const prefixCacheRamBudget = Number(runtime?.prefix_cache_total_hot_max_bytes ?? runtime?.prefix_cache_max_bytes ?? 0);
  const prefixCacheQueries = Number(runtime?.prefix_cache_queries || 0);
  const prefixCacheHits = Number(runtime?.prefix_cache_hits || 0);
  const prefixCacheSnapshots = Number(runtime?.prefix_cache_snapshots || 0);
  const prefixCacheBytes = Number(runtime?.prefix_cache_bytes || 0);
  const prefixCacheDiskBytes = Number(runtime?.prefix_cache_total_disk_bytes ?? runtime?.prefix_cache_disk_bytes ?? 0);
  const prefixCacheDiskBudget = Number(runtime?.prefix_cache_total_disk_max_bytes ?? runtime?.prefix_cache_disk_max_bytes ?? 0);
  const prefixCacheHotBytes = Number(runtime?.prefix_cache_total_hot_bytes ?? runtime?.prefix_cache_hot_bytes ?? prefixCacheBytes);
  const prefixCacheHitRate =
    prefixCacheQueries > 0 ? (prefixCacheHits / prefixCacheQueries) * 100 : 0;
  const prefixCacheHotOnly = runtime?.prefix_cache_mode === 'single_device_hot_prefix';
  const prefixCachePoolReported = typeof runtime?.prefix_cache_total_disk_max_bytes === 'number';
  const prefixCachePersistent = prefixCachePoolReported || typeof runtime?.prefix_cache_max_bytes === 'number';
  const prefixCacheSupported = prefixCachePoolReported || (runtime?.prefix_cache_supported !== undefined
    ? Number(runtime.prefix_cache_supported) > 0
    : prefixCachePersistent || prefixCacheHotOnly);
  const prefixCacheUnavailableReason = Number(runtime?.prefix_cache_disabled_reason) === 2
    ? tr(
        '连续批处理模式暂不支持 Session KV 缓存',
        'Session KV cache is unavailable with continuous batching',
      )
    : tr(
        '当前模型不支持 Session KV 缓存',
        'The current model does not support Session KV cache',
      );
  return <>
      <SectionLabel title={tr('前缀缓存', 'Prefix cache')} />
      {prefixCacheSupported ? (
        <TMPanel className="cache-panel">
          <div className="panel-heading">
            <div>
              <h2>{tr('前缀缓存概况', 'Prefix cache overview')}</h2>
              <p>
                {prefixCacheHotOnly
                  ? tr(
                      '当前进程内的单条设备热前缀；切换会话或编辑 prompt 会重新 prefill',
                      'One device-hot prefix in this process; switching sessions or editing the prompt triggers a fresh prefill',
                    )
                  : vramPrefix ? tr('显存热缓存、低优先级 RAM 前缀缓存与可选 SSD 冷层', 'VRAM hot cache, low-priority RAM prefixes and optional SSD cold tier') : tr(
                      'RAM 热缓存与可跨重启复用的 SSD 前缀块',
                      'RAM hot cache with persistent SSD prefix blocks',
                    )}
              </p>
            </div>
            <b>
              {prefixCacheQueries > 0
                ? `${formatNumber(prefixCacheHitRate, 1)}% hit`
                : tr('暂无查询', 'No queries')}
            </b>
          </div>
          {prefixCacheHotOnly ? (
            <>
              <UsageBar
                label={separate ? tr('显存热前缀', 'VRAM hot prefix') : tr('RAM 热前缀', 'RAM hot prefix')}
                used={prefixCacheHotBytes}
                total={Math.max(prefixCacheHotBytes, prefixCacheRamBudget)}
              />
              <div className="cache-stats">
                <div>
                  <span>{tr('活动会话', 'Active sessions')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_sessions)}</strong>
                </div>
                <div>
                  <span>{tr('设备热前缀', 'Device-hot prefix')}</span>
                  <strong>{formatNumber(prefixCacheSnapshots)}</strong>
                  <small>{tr('单物理快照', 'one physical snapshot')}</small>
                </div>
                <div>
                  <span>{tr('缓存 tokens', 'Cached tokens')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_tokens)}</strong>
                </div>
                <div>
                  <span>{tr('复用 tokens', 'Reused tokens')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_hit_tokens)}</strong>
                </div>
                <div>
                  <span>{tr('命中 / 查询', 'Hits / queries')}</span>
                  <strong>
                    {formatNumber(prefixCacheHits)} / {formatNumber(prefixCacheQueries)}
                  </strong>
                </div>
                <div>
                  <span>{tr('持久性', 'Persistence')}</span>
                  <strong>{tr('进程生命周期', 'Process lifetime')}</strong>
                  <small>{tr('不落盘', 'not stored on disk')}</small>
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="cache-usage-bars">
                {(!separate || vramPrefix) && <UsageBar
                  label={vramPrefix ? tr('显存热缓存总占用', 'Total VRAM hot cache') : tr('RAM 热缓存总占用', 'Total RAM hot cache')}
                  used={prefixCacheHotBytes}
                  total={Math.max(prefixCacheHotBytes, prefixCacheRamBudget)}
                />}
                {separate && <UsageBar
                  label={tr('RAM 前缀缓存总占用', 'Total RAM prefix cache')}
                  used={runtime?.prefix_cache_total_ram_bytes ?? runtime?.prefix_cache_ram_bytes ?? 0}
                  total={Math.max(runtime?.prefix_cache_total_ram_bytes ?? runtime?.prefix_cache_ram_bytes ?? 0,
                    runtime?.prefix_cache_total_ram_max_bytes ?? runtime?.prefix_cache_ram_max_bytes ?? 0)}
                />}
                <UsageBar
                  label={tr('SSD 前缀缓存总占用', 'Total SSD prefix cache')}
                  used={prefixCacheDiskBytes}
                  total={Math.max(prefixCacheDiskBudget, prefixCacheDiskBytes)}
                />
              </div>
              <div className="cache-stats">
                <div>
                  <span>{tr('活动会话', 'Active sessions')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_sessions)}</strong>
                </div>
                <div>
                  <span>{tr('SSD 块', 'SSD blocks')}</span>
                  <strong>
                    {formatNumber(runtime?.prefix_cache_total_disk_blocks ?? runtime?.prefix_cache_disk_blocks ?? prefixCacheSnapshots)}
                  </strong>
                </div>
                <div>
                  <span>{tr('复用 tokens', 'Reused tokens')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_hit_tokens)}</strong>
                </div>
                <div>
                  <span>{tr('SSD 占用', 'SSD usage')}</span>
                  <strong>{formatNumber(prefixCacheDiskBytes / 2 ** 30, 2)} GiB</strong>
                  <small>
                    {prefixCacheDiskBudget > 0
                      ? `${formatNumber(prefixCacheDiskBudget / 2 ** 30, 0)} GiB ${tr('上限', 'limit')}`
                      : ''}
                  </small>
                </div>
                <div>
                  <span>{separate ? tr('显存热层', 'VRAM hot tier') : tr('RAM 热层', 'RAM hot tier')}</span>
                  <strong>{formatNumber(prefixCacheHotBytes / 2 ** 20, 1)} MiB</strong>
                  <small>{formatNumber(runtime?.prefix_cache_hot_blocks)} blocks</small>
                </div>
                <div>
                  <span>{tr('待写入', 'Pending writes')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_pending_writes)}</strong>
                  <small>
                    {formatNumber(Number(runtime?.prefix_cache_pending_bytes || 0) / 2 ** 20, 1)} /{' '}
                    {formatNumber(
                      Number(runtime?.prefix_cache_pending_max_bytes || 0) / 2 ** 20,
                      0,
                    )}{' '}
                    MiB · {formatNumber(runtime?.prefix_cache_deduplicated_writes)} deduplicated
                  </small>
                </div>
                <div>
                  <span>{tr('SSD 命中', 'SSD hits')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_disk_hits)}</strong>
                  <small>{formatNumber(separate && !vramPrefix ? runtime?.prefix_cache_ram_hits : runtime?.prefix_cache_hot_hits)} {vramPrefix ? 'VRAM' : 'RAM'} hits</small>
                  {separate && <small>{formatNumber(runtime?.prefix_cache_ram_hits)} RAM hits</small>}
                </div>
                <div>
                  <span>{tr('回收', 'Evictions')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_evictions)}</strong>
                  <small>{formatNumber(runtime?.prefix_cache_corrupt_blocks)} corrupt</small>
                </div>
              </div>
            </>
          )}
          {(onClear && prefixCacheSnapshots > 0 || onTrim) && (
            <div className="cache-panel-actions">
              {onClear && prefixCacheSnapshots > 0 && <button
                className="cache-clear-button"
                disabled={busy || Number(runtime?.active_requests || 0) > 0}
                onClick={onClear}
                type="button"
              >
                <Icon name="trash" size={14} />
                {tr('清除此模型的前缀缓存', 'Clear this model’s prefix cache')}
              </button>}
              {onTrim && <button disabled={busy || Number(runtime?.active_requests || 0) > 0} onClick={onTrim} type="button">{tr('缩减此模型的热缓存', 'Trim this model’s hot cache')}</button>}
            </div>
          )}
        </TMPanel>
      ) : (
        <EmptyPanel
          icon="settings"
          title={
            runtime
              ? tr('Session KV 缓存不可用', 'Session KV cache unavailable')
              : tr('Runtime 诊断未启用', 'Runtime diagnostics are offline')
          }
          message={
            runtime
              ? prefixCacheUnavailableReason
              : tr(
                  '加载模型后可查看内存与前缀缓存状态。',
                  'Load a model to inspect memory and prefix-cache state.',
                )
          }
        />
      )}
  </>;
}
