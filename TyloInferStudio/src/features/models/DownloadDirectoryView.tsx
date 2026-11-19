import type { DownloadDirectoryStatus } from '../../shared/api/resources/models';
import type { MouseEventHandler } from 'react';
import { Icon } from '../../app/display';
import { useSettings } from '../settings/SettingsProvider';
import { ModelMemoryPressure } from './ModelMemoryPressure';
export function DownloadDirectoryView({ directory, storage, saving, savingReclaim, onBrowse, onReclaim }: {
  directory: string | null; storage: DownloadDirectoryStatus | null; saving: boolean; savingReclaim: boolean;
  onBrowse: MouseEventHandler<HTMLButtonElement>; onReclaim: (enabled: boolean) => void;
}) {
  const { tr } = useSettings();
  const free = storage?.disk_free_bytes;
  const total = storage?.disk_total_bytes;
  const known = free != null && total != null && Number.isFinite(free) && Number.isFinite(total) && free >= 0 && total > 0;

  return (
    <div className="download-directory-row">
    <div className="download-directory-control">
      <span>{tr('下载目录', 'Download folder')}</span>
      <code title={directory || undefined}>{directory || '--'}</code>
      <button type="button" disabled={!directory || saving} onClick={onBrowse}><Icon name="folder" size={14} />{tr('更改', 'Change')}</button>
    </div>
    <div className="download-storage">
      <div className="download-storage-summary"><span>{tr('SSD 剩余', 'SSD free')}</span>
        <span>{known ? `${(free / 2 ** 30).toFixed(1)} / ${(total / 2 ** 30).toFixed(1)} GiB` : '--'}</span></div>
      <ModelMemoryPressure required={known ? free : null} available={known ? total : null}
        label={tr('SSD 剩余空间', 'SSD free space')} emptyLabel="--" />
      <label className="download-prefix-reclaim" title={tr('将同盘可回收的 SSD Prefix Cache 计入可下载空间；下载时缩减其预算并按 LRU 驱逐缓存。', 'Include reclaimable SSD prefix cache on this drive in available download space. Downloads reduce its budget and evict least recently used blocks.')}>
        <input type="checkbox" checked={storage?.evict_prefix_cache ?? false} disabled={!storage || saving || savingReclaim}
          onChange={event => onReclaim(event.target.checked)} />
        {tr('驱逐 SSD Prefix 预算', 'Reclaim SSD prefix budget')}
      </label>
    </div>
    </div>
  );
}
