import { Icon } from '../../app/display';
import { formatBytes } from '../../app/formatters';
import type { ModelArtifact, RuntimeInstance } from '../../shared/api/types';
export function ModelLibrarySummary({ artifacts, instances, folderPath, tr }: {
  artifacts: ModelArtifact[]; instances: RuntimeInstance[]; folderPath?: string | null; tr(zh: string, en: string): string;
}) {
  const totalBytes = artifacts.reduce((sum, artifact) => sum + (artifact.owned_file_bytes ?? artifact.total_bytes), 0);
  const modelCount = artifacts.filter(artifact => !artifact.asset_kind || artifact.asset_kind === 'model').length;
  const loadedCount = new Set(instances.filter(instance => instance.state === 'ready' || instance.state === 'busy').map(instance => instance.model)).size;
  return <>
    <div className="model-folder-location"><Icon name="folder" /><span>{tr('模型文件夹', 'Model folder')}</span><code title={folderPath || undefined}>{folderPath || tr('尚未选择', 'Not selected')}</code></div>
    <div className="model-library-summary">
      <span>{tr('已加载', 'Loaded')} <strong>{loadedCount}</strong></span><span>{tr('本地模型', 'Local models')} <strong>{modelCount}</strong></span>
      <span>{tr('文件总大小', 'Total file size')} <strong>{totalBytes === 0 ? '0 B' : formatBytes(totalBytes).replace(/\b(KB|MB|GB|TB)\b/g, unit => `${unit[0]}iB`)}</strong></span>
    </div>
  </>;
}
