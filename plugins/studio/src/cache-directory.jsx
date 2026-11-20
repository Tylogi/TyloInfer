import React, { useEffect, useRef, useState } from 'react';
import { ModelDirectoryDialog } from '../../../studio/web/src/features/models/ModelDirectoryDialog.tsx';
import { Dialog } from './shared-primitives.jsx';
import { terminalStates } from './model-api.js';

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
    <Dialog open={Boolean(confirmation)} onOpenChange={value => { if (!value) setConfirmation(null); }} title={tr('更改全部模型的缓存目录', 'Change cache folder for all loaded models')} closeLabel={tr('关闭', 'Close')} returnFocusRef={trigger}>
      <div className="tylo-model-confirm"><p>{tr('当前目录', 'Current folder')}: {currentDirectory || '—'}</p><strong>{confirmation?.path}</strong>
        <p>{tr('范围：全部已加载模型。全局内存规划可能先卸载再重载所有模型，并清空活动 KV；进行中的请求不会被自动取消。此操作不是普通路径保存，失败也可能已有部分副作用。', 'Scope: all loaded models. Global memory planning may unload and reload every model and clear active KV. Active requests are not cancelled automatically. This is not a simple path save; failures may still have partial side effects.')}</p>
        <p>{tr('当前已加载模型', 'Currently loaded models')}: {(inventory?.instances || []).filter(item => ['ready', 'busy', 'loading'].includes(item.state)).map(item => item.model).join(' · ') || tr('无', 'None')}</p>
        <p>{tr('仅提交所选核心目录 ID，不修改预算默认值；实际目录可写性与模型恢复由核心校验。受理后自动跟踪内存规划作业，不会重发。', 'Submit only the selected core directory ID without resetting budget defaults. The core checks writability and model recovery. Native memory-plan progress is tracked after acceptance without replaying the request.')}</p>
        <div className="tylo-model-confirm-actions"><button type="button" onClick={() => setConfirmation(null)}>{tr('取消', 'Cancel')}</button><button type="button" className="danger" disabled={blocked} onClick={() => void confirm()}>{tr('确认更改全部模型目录', 'Confirm global folder change')}</button></div>
      </div>
    </Dialog>
  </>;
}
