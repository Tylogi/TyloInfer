import type { MouseEvent } from 'react';
import { Icon } from '@shared/ui/display';
import { formatBytes } from '@shared/formatters';
import type { ModelArtifact, RuntimeInstance } from '@shared/types/api';
export function ModelLibrarySummary({ artifacts, instances, folderPath, onChooseFolder, busy, tr }: {
  artifacts: ModelArtifact[]; instances: RuntimeInstance[]; folderPath?: string | null; tr(zh: string, en: string): string;
  onChooseFolder?: (event: MouseEvent<HTMLButtonElement>) => void; busy?: boolean;
}) {
  const totalBytes = artifacts.reduce((sum, artifact) => sum + (artifact.owned_file_bytes ?? artifact.total_bytes), 0);
  const modelCount = artifacts.filter(artifact => !artifact.asset_kind || artifact.asset_kind === 'model').length;
  const loadedCount = new Set(instances.filter(instance => instance.state === 'ready' || instance.state === 'busy').map(instance => instance.model)).size;
  return <>
    <div className="model-folder-location">
      <Icon name="folder" /><span>{tr('模型文件夹', 'Model folder')}</span>
      <code title={folderPath || undefined}>{folderPath || tr('尚未选择', 'Not selected')}</code>
      {onChooseFolder && <button type="button" disabled={busy} onClick={onChooseFolder}
        aria-label={tr('选择模型文件夹', 'Choose model folder')}>{folderPath ? tr('更改', 'Change') : tr('选择', 'Choose')}</button>}
    </div>
    <div className="model-library-summary">
      <span>{tr('已加载', 'Loaded')} <strong>{loadedCount}</strong></span><span>{tr('本地模型', 'Local models')} <strong>{modelCount}</strong></span>
      <span>{tr('文件总大小', 'Total file size')} <strong>{totalBytes === 0 ? '0 B' : formatBytes(totalBytes).replace(/\b(KB|MB|GB|TB)\b/g, unit => `${unit[0]}iB`)}</strong></span>
    </div>
  </>;
}
