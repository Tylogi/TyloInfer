import React, { useEffect, useRef, useState } from 'react';
import { DownloadDirectoryView } from '../../../TyloInferStudio/src/features/models/DownloadDirectoryView.tsx';
import { ModelDirectoryDialog } from '../../../TyloInferStudio/src/features/models/ModelDirectoryDialog.tsx';
import { ConfirmationDialog } from './shared-primitives.jsx';
import { CoreError, Facts } from './page-kit.jsx';

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
    <ConfirmationDialog open={Boolean(confirm)} tr={tr} title={tr('保存下载设置', 'Save download settings')}
      description={tr('设置适用于新下载。已有文件和排队中的任务保留原目录。', 'These settings apply to new downloads. Existing files and queued jobs keep their current locations.')}
      confirmLabel={tr('保存', 'Save')} onCancel={() => setConfirm(null)} onConfirm={save} disabled={busy} returnFocusRef={trigger}>
      <Facts entries={[[tr('下载目录', 'Download folder'), confirm?.path], [tr('回收 SSD 前缀缓存', 'Reclaim SSD prefix caches'), confirm?.reclaim ? tr('允许', 'Allowed') : tr('关闭', 'Off')]]} />
      {confirm?.reclaim && <p>{tr('同一磁盘空间不足时，后续下载可缩减前缀预算并清理 SSD 缓存块。', 'Later downloads may reduce prefix budgets and clear SSD blocks when disk space is low.')}</p>}
    </ConfirmationDialog>
  </>;
}
