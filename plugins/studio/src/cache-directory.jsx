import React, { useEffect, useRef, useState } from 'react';
import { ModelDirectoryDialog } from '../../../studio/web/src/features/models/ModelDirectoryDialog.tsx';
import { ConfirmationDialog } from './shared-primitives.jsx';
import { terminalStates } from './model-api.js';
import { Facts } from './page-kit.jsx';

export function CacheDirectoryControl({ api, mutation, tr, refreshKey, inventory, currentDirectory }) {
  const [open, setOpen] = useState(false);
  const [directoryId, setDirectoryId] = useState(null);
  const [browser, setBrowser] = useState({ value: null, loading: false, error: null });
  const [confirmation, setConfirmation] = useState(null);
  const trigger = useRef(null);
  const blocked = mutation.busy || mutation.unknown || Boolean(mutation.operation && !terminalStates.has(mutation.operation.status));
  useEffect(() => { setOpen(false); setConfirmation(null); }, [refreshKey]);
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController(); setBrowser({ value: null, loading: true, error: null });
    void api.read('model_directories', directoryId ? { directory_id: directoryId } : {}, controller.signal).then(value => {
      if (!Array.isArray(value?.data) || value.current_id != null && !/^[0-9a-f]{32}$/.test(value.current_id)) throw new Error('Invalid core directory metadata');
      if (!controller.signal.aborted) setBrowser({ value, loading: false, error: null });
    }).catch(error => { if (!controller.signal.aborted) setBrowser({ value: null, loading: false, error }); });
    return () => controller.abort();
  }, [open, directoryId, api]);
  function choose() {
    const value = browser.value;
    if (blocked || browser.loading || !/^[0-9a-f]{32}$/.test(value?.current_id) || typeof value.current_path !== 'string' || !value.current_path) return;
    setConfirmation({ directory_id: value.current_id, path: value.current_path }); setOpen(false);
  }
  async function confirm() {
    const target = confirmation; if (!target || blocked) return; setConfirmation(null);
    await mutation.run(signal => api.configureCacheDirectory(target.directory_id, signal), 'memory');
  }
  return <><div className="server-row-actions prefix-directory-controls"><button ref={trigger} type="button" disabled={blocked} onClick={() => { setDirectoryId(null); setOpen(true); }}>{tr('选择文件夹', 'Choose folder')}</button></div>
    <ModelDirectoryDialog title={tr('选择前缀缓存目录', 'Choose prefix cache folder')} navigation="id" showNativeOpen={false} catalog={{
      modelBrowser: browser.value, modelBrowserOpen: open, modelBrowserError: browser.error?.message || '', modelBrowserLoading: browser.loading,
      modelFilesMode: false, setModelBrowserOpen: setOpen, modelBrowserTriggerRef: trigger, busy: blocked || browser.loading,
      jumpToModelDirectory: event => event.preventDefault(), openModelDirectory: id => setDirectoryId(id || null),
      modelDirectoryPath: browser.value?.current_path || browser.value?.current_name || '', setModelDirectoryPath() {}, registerCurrentModelDirectory: choose, openCurrentDirectory() {},
    }} />
    <ConfirmationDialog open={Boolean(confirmation)} tr={tr} title={tr('更改前缀缓存目录', 'Change prefix cache folder')}
      description={tr('更改会应用到全部模型，可能重载模型并清空活动 KV。', 'This change applies to every model and may reload models and clear active KV.')}
      returnFocusRef={trigger} confirmLabel={tr('更改目录', 'Change folder')} onCancel={() => setConfirmation(null)} onConfirm={confirm} disabled={blocked} danger>
      <Facts entries={[[tr('当前目录', 'Current folder'), currentDirectory || null], [tr('新目录', 'New folder'), confirmation?.path],
        [tr('已加载的模型', 'Loaded models'), (inventory?.instances || []).filter(item => ['ready', 'busy', 'loading'].includes(item.state)).map(item => item.model).join(' · ') || tr('无', 'None')]]} />
    </ConfirmationDialog>
  </>;
}
