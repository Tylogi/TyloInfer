import React, { useEffect, useState } from 'react';
import { ModelDirectoryDialog } from '../../../studio/web/src/features/models/ModelDirectoryDialog.tsx';

export function ModelRegistration({ api, mutation, refreshKey, open, onOpenChange, triggerRef, modelId, onRegistered }) {
  const [directoryId, setDirectoryId] = useState('');
  const [state, setState] = useState({ loading: false, value: null, error: null });
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setState({ loading: true, value: null, error: null });
    const request = modelId && !directoryId ? api.directory(modelId, controller.signal) : api.read('model_directories', directoryId ? { directory_id: directoryId } : {}, controller.signal);
    void request.then(value => {
      if (!Array.isArray(value.data)) throw new Error('Invalid directory list');
      if (!controller.signal.aborted) setState({ loading: false, value, error: null });
    }).catch(error => { if (!controller.signal.aborted) setState({ loading: false, value: null, error }); });
    return () => controller.abort();
  }, [api, directoryId, refreshKey, open, modelId]);
  const choose = async () => {
    const id = state.value?.current_id;
    if (!id || state.loading || mutation.busy || mutation.unknown) return;
    const path = state.value?.current_path;
    const result = await mutation.run(signal => api.register(id, signal));
    if (result) { if (typeof path === 'string') onRegistered?.(path); onOpenChange(false); }
  };
  const catalog = {
    modelBrowser: state.value,
    modelBrowserOpen: open,
    modelBrowserError: state.error?.message || '',
    modelBrowserLoading: state.loading,
    modelFilesMode: Boolean(modelId),
    setModelBrowserOpen: onOpenChange,
    modelBrowserTriggerRef: triggerRef,
    busy: state.loading || mutation.busy || mutation.unknown,
    jumpToModelDirectory: event => event.preventDefault(),
    openModelDirectory: id => setDirectoryId(id || ''),
    modelDirectoryPath: state.value?.current_path || state.value?.current_name || '',
    setModelDirectoryPath() {},
    registerCurrentModelDirectory: choose,
    openCurrentDirectory() {},
  };
  return <ModelDirectoryDialog catalog={catalog} navigation="id" showNativeOpen={false} />;
}
