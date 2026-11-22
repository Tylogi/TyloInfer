import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router';
import { useSettings } from '../settings/SettingsProvider';
import { Icon, ScreenHeader } from '../../app/display';
import { useModelCatalog } from './useModelCatalog';
import { ModelDirectoryDialog } from './ModelDirectoryDialog';
import { LoadedModels } from './LoadedModels';
import { ModelSettingsDialog } from './ModelSettingsDialog';
import { LocalCheckpoints } from './LocalCheckpoints';
import { ModelLibrarySummary } from './ModelLibrarySummary';
import '../../shared/styles/configuration-layout.css';

export function ModelsPage() {
  const catalog = useModelCatalog();
  const { tr } = useSettings();
  const { hash } = useLocation();
  const checkpoints = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (hash === '#local-checkpoints') checkpoints.current?.scrollIntoView({ block: 'start' });
  }, [hash]);
  const { artifacts, busy,
    openStudioPage, chooseModelDirectory, modelFolderPath } = catalog;
  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={tr('模型', 'Models')}
        subtitle={tr('管理本地模型与运行实例。', 'Manage local models and runtime instances.')}
        trailing={
          <>
            <button onClick={() => openStudioPage('lab', 'models')} type="button">
              <Icon name="download" size={14} />{tr('模型下载', 'Model downloads')}
            </button>
            <button className="primary" disabled={busy}
              onClick={() => void chooseModelDirectory()} type="button">
              <Icon name="folder" size={14} />{tr('添加模型', 'Add model')}
            </button>
          </>
        }
      />
      <ModelLibrarySummary artifacts={artifacts} instances={catalog.instances || []} folderPath={modelFolderPath} busy={busy}
        onChooseFolder={() => void chooseModelDirectory()} tr={tr} />
      <section id="local-checkpoints" ref={checkpoints}>
        <LocalCheckpoints catalog={catalog} />
      </section>
      {catalog.instances?.some(instance => !artifacts.some(artifact => artifact.name === instance.model)) && <LoadedModels catalog={catalog} unlistedOnly />}
      <ModelSettingsDialog key={`${catalog.connectionRevision}:${catalog.settingsModel ?? ''}`} catalog={catalog} />
      <ModelDirectoryDialog catalog={catalog} />
    </section>
  );
}
