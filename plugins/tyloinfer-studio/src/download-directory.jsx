import React, { useEffect, useRef, useState } from 'react';
import { DownloadDirectoryView } from '../../../TyloInferStudio/src/features/models/DownloadDirectoryView.tsx';
import { ModelDirectoryDialog } from '../../../TyloInferStudio/src/features/models/ModelDirectoryDialog.tsx';
import { Dialog } from './shared-primitives.jsx';
import { CoreError } from './page-kit.jsx';

export function DownloadDirectory({ entry, api, mutation, tr, refreshKey }) {
  const [open, setOpen] = useState(false);
  const [directoryId, setDirectoryId] = useState(null);
  const [browser, setBrowser] = useState({ value: null, loading: false, error: null });
  const [confirm, setConfirm] = useState(null);
  const trigger = useRef(null);
  const storage = entry?.value || null;
  const busy = mutation.busy || mutation.unknown;
  useEffect(() => { setOpen(false); setConfirm(null); }, [refreshKey]);
  useEffect(() => { setConfirm(null); }, [storage?.actual_directory, storage?.evict_prefix_cache]);
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController(); setBrowser({ value: null, loading: true, error: null });
    void api.read('model_directories', directoryId ? { directory_id: directoryId } : {}, controller.signal).then(value => {
      if (!controller.signal.aborted) setBrowser({ value, loading: false, error: null });
    }).catch(error => { if (!controller.signal.aborted) setBrowser({ value: null, loading: false, error }); });
    return () => controller.abort();
  }, [open, directoryId, api]);
  function choose() {
    if (!browser.value?.current_id || busy || browser.loading) return;
    setConfirm({ request: { directory_id: browser.value.current_id }, path: browser.value.current_path || browser.value.current_name, reclaim: storage?.evict_prefix_cache });
    setOpen(false);
  }
  async function save() {
    const target = confirm; if (!target || busy) return; setConfirm(null);
    await mutation.run(signal => api.configureDownloadDirectory(target.request, signal));
  }
  return <>
    {entry?.error && <CoreError error={entry.error} />}
    <DownloadDirectoryView directory={storage?.actual_directory || null} storage={storage} saving={busy} savingReclaim={busy}
      onBrowse={event => { trigger.current = event.currentTarget; setDirectoryId(null); setOpen(true); }}
      onReclaim={enabled => { if (!storage || busy) return; setConfirm({ request: { evict_prefix_cache: enabled }, path: storage.actual_directory, reclaim: enabled }); }} />
    <ModelDirectoryDialog title={tr('选择下载文件夹', 'Choose download folder')} navigation="id" showNativeOpen={false} catalog={{
      modelBrowser: browser.value, modelBrowserOpen: open, modelBrowserError: browser.error?.message || '', modelBrowserLoading: browser.loading,
      modelFilesMode: false, setModelBrowserOpen: setOpen, modelBrowserTriggerRef: trigger, busy: busy || browser.loading,
      jumpToModelDirectory: event => event.preventDefault(), openModelDirectory: id => setDirectoryId(id || null),
      modelDirectoryPath: browser.value?.current_path || browser.value?.current_name || '', setModelDirectoryPath() {},
      registerCurrentModelDirectory: choose, openCurrentDirectory() {},
    }} />
    <Dialog open={Boolean(confirm)} onOpenChange={value => { if (!value) setConfirm(null); }} title={tr('保存下载设置', 'Save download settings')} closeLabel={tr('关闭', 'Close')} returnFocusRef={trigger}>
      <div className="tylo-model-confirm"><strong>{confirm?.path}</strong><p>{confirm?.reclaim ? tr('允许后续下载在同盘空间不足时回收 SSD 前缀缓存，可能缩减前缀预算并删除离线块。保存时不立即清理或下载。', 'Allow later downloads to reclaim SSD prefix caches when space is low. This can reduce prefix budgets and remove offline blocks. Saving does not immediately purge or download.') : tr('后续下载不自动回收 SSD 前缀缓存。', 'Later downloads will not automatically reclaim SSD prefix caches.')}</p><p>{tr('不迁移已有文件或已排队任务；新下载固定使用确认时的目录。', 'Existing files and queued jobs are not moved. New downloads use their confirmed directory.')}</p>
        <div className="tylo-model-confirm-actions"><button type="button" onClick={() => setConfirm(null)}>{tr('取消', 'Cancel')}</button><button type="button" className="primary" disabled={busy} onClick={() => void save()}>{tr('保存', 'Save')}</button></div>
      </div>
    </Dialog>
  </>;
}
