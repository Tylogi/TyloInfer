
import { Dialog } from '../../shared/ui/Dialog';
import { Icon } from '../../app/display';
import { useSettings } from '../settings/SettingsProvider';
import type { useModelCatalog } from './useModelCatalog';
import { formatBytes } from '../../app/formatters';
import type { ModelDirectoryEntry } from '../../shared/api/contracts/models';
import { ArrowUpIcon, HardDriveIcon, HouseIcon } from '@phosphor-icons/react';
import { useEffect, useRef } from 'react';


type DirectoryDialogState = Pick<ReturnType<typeof useModelCatalog>, 'modelBrowser' | 'modelBrowserOpen' | 'modelBrowserError' | 'modelBrowserLoading' | 'modelFilesMode' | 'setModelBrowserOpen' | 'modelBrowserTriggerRef' | 'busy' | 'jumpToModelDirectory' | 'openModelDirectory' | 'modelDirectoryPath' | 'setModelDirectoryPath' | 'registerCurrentModelDirectory' | 'openCurrentDirectory'>;

export function ModelDirectoryDialog({ catalog, title, navigation = 'path', showNativeOpen = true, registerModelFile, registering = false }: {
  catalog: DirectoryDialogState;
  title?: string;
  navigation?: 'path' | 'id';
  showNativeOpen?: boolean;
  registerModelFile?: (name: string) => Promise<void> | void;
  registering?: boolean;
}) {
  const { tr } = useSettings();
  const {
    modelBrowser,
    modelBrowserOpen,
    modelBrowserError,
    modelBrowserLoading,
    modelFilesMode,
    setModelBrowserOpen,
    modelBrowserTriggerRef,
    busy,
    jumpToModelDirectory,
    openModelDirectory,
    modelDirectoryPath,
    setModelDirectoryPath,
    registerCurrentModelDirectory,
    openCurrentDirectory,
  } = catalog;
  const hasUnvisitedPath = navigation === 'path' && Boolean(modelBrowser?.current_path) &&
    modelDirectoryPath.trim() !== modelBrowser?.current_path;
  const atRoot = modelBrowser != null && !modelBrowser.current_id;
  const list = useRef<HTMLDivElement>(null);
  useEffect(() => { if (list.current) list.current.scrollTop = 0; }, [modelBrowserOpen, modelBrowser?.current_id]);
  const directoryRow = (directory: ModelDirectoryEntry, root = false) => (
    <button disabled={busy} key={directory.id} onClick={() => void openModelDirectory(directory.id)} type="button">
      {root ? <HardDriveIcon size={16} aria-hidden="true" /> : <Icon name="folder" />}
      <span>{directory.name}</span>
      {directory.model_file_count > 0 && <b>{directory.model_file_count} {tr('个模型', 'models')}</b>}
    </button>
  );
  return (
    <Dialog
      open={modelBrowserOpen}
      onOpenChange={setModelBrowserOpen}
      title={title || (modelFilesMode ? tr('模型文件', 'Model files') : registerModelFile ? tr('导入模型', 'Import models') : tr('选择模型文件夹', 'Choose model folder'))}
      description={modelFilesMode ? tr('模型所在的实际目录。', 'The directory containing this model’s files.') : tr(
        '浏览 TyloInfer Server 所在设备上的文件夹。',
        'Browse folders on the TyloInfer Server host.',
      )}
      closeLabel={tr('关闭', 'Close')}
      className="model-browser-dialog"
      returnFocusRef={modelBrowserTriggerRef}
    >
      <form className="model-browser-location" onSubmit={jumpToModelDirectory}>
        <div className="model-browser-navigation">
          <button disabled={busy || (atRoot && !modelBrowserError)}
            onClick={() => void openModelDirectory(null)} type="button">
            <HouseIcon size={14} aria-hidden="true" />{tr('根目录', 'Roots')}
          </button>
          <button
            disabled={busy || !modelBrowser?.current_id}
            onClick={() => void openModelDirectory(modelBrowser?.parent_id ?? null)}
            type="button"
          >
            <ArrowUpIcon size={14} aria-hidden="true" />{tr('上一级', 'Up')}
          </button>
        </div>
        <input
          aria-label={tr('当前目录', 'Current directory')}
          disabled={busy}
          readOnly={navigation === 'id'}
          onChange={(event) => setModelDirectoryPath(event.target.value)}
          placeholder={navigation === 'path' ? tr('输入服务器上的完整目录', 'Enter a full directory on the server') : tr('根目录', 'Roots')}
          spellCheck={false}
          value={modelDirectoryPath}
        />
        {navigation === 'path' && <button disabled={busy || !modelDirectoryPath.trim()} type="submit">
          {tr('前往', 'Go')}
        </button>}
        {modelBrowser?.current_id && <span>{modelBrowser.model_file_count} {tr('个模型', 'models')}</span>}
      </form>
      <div ref={list} className="model-browser-list" aria-busy={modelBrowserLoading}>
        {modelBrowserLoading && <p role="status">{tr('正在读取目录', 'Loading folder')}</p>}
        {registering && <p role="status">{tr('正在导入模型…', 'Importing models…')}</p>}
        {modelBrowserError && <p role="alert">{modelBrowserError}</p>}
        {atRoot && Boolean(modelBrowser?.shortcuts?.length) && <>
          <h3>{tr('常用目录', 'Quick access')}</h3>
          {modelBrowser?.shortcuts?.map(directory => directoryRow(directory))}
        </>}
        {atRoot && Boolean(modelBrowser?.data.length) && <h3>{tr('磁盘和根目录', 'Drives and roots')}</h3>}
        {modelBrowser?.data.map(directory => directoryRow(directory, atRoot))}
        {modelBrowser?.files?.map((file) => <div className={registerModelFile ? 'model-browser-file has-import' : 'model-browser-file'} key={file.name}>
          <Icon name="file" />
          <span title={file.name}>{file.name}</span>
          <small>{file.byte_size === 0 ? '0 B' : formatBytes(file.byte_size).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`)}</small>
          {registerModelFile && <button type="button" disabled={busy || hasUnvisitedPath || Boolean(modelBrowserError)}
            aria-label={`${tr('导入', 'Import')} ${file.name}`} onClick={() => void registerModelFile(file.name)}>
            {tr('导入', 'Import')}
          </button>}
        </div>)}
        {modelBrowser && modelBrowser.data.length === 0 && !modelBrowser.shortcuts?.length && !modelBrowser.files?.length && !modelBrowser.model_file_count && !modelBrowserLoading && !modelBrowserError && (
          <p>{tr('这个文件夹中没有子文件夹或模型文件。', 'This folder has no subfolders or model files.')}</p>
        )}
      </div>
      <footer className="model-browser-footer">
        {showNativeOpen && <button className="model-browser-open-finder" disabled={busy || !modelBrowser?.current_id || !(modelBrowser.can_open_directory ?? modelBrowser.can_open_in_finder) || Boolean(modelBrowserError)}
          onClick={() => void openCurrentDirectory()} type="button"
          title={modelBrowser?.can_open_in_finder ? tr('在访达中打开', 'Open in Finder') : tr('在文件管理器中打开', 'Open in file manager')}>
          <Icon name="folder" size={13} />{tr('打开目录', 'Open folder')}
        </button>}
        <div className="model-browser-footer-actions">
          <button onClick={() => setModelBrowserOpen(false)} type="button">
            {tr('取消', 'Cancel')}
          </button>
          {!modelFilesMode && <button
            className="primary"
            disabled={busy || hasUnvisitedPath || !modelBrowser?.current_id || Boolean(modelBrowserError)}
            onClick={() => void registerCurrentModelDirectory()}
            type="button"
          >
            {registerModelFile && Boolean(modelBrowser?.model_file_count) ? tr('导入目录', 'Import folder') : tr('选定目录', 'Select folder')}
          </button>}
        </div>
      </footer>
    </Dialog>
  );
}
