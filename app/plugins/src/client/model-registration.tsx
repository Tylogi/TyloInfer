import React, { useEffect, useState } from 'react';
import { ModelDirectoryDialog } from '@shared/features/models/ModelDirectoryDialog.tsx';

export function ModelRegistration({ api, mutation, refreshKey, open, onOpenChange, triggerRef, modelId, startPath, onRegistered }) {
  const [location, setLocation] = useState(null);
  const [directoryPath, setDirectoryPath] = useState('');
  const [state, setState] = useState({ loading: false, value: null, error: null });
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setState(previous => ({ ...previous, loading: true, error: null }));
    const path = location?.path ?? (!location ? startPath : null);
    const request = path ? api.browseDirectory(path, controller.signal)
      : modelId && !location ? api.directory(modelId, controller.signal)
      : api.read('model_directories', location?.id ? { directory_id: location.id } : {}, controller.signal);
    void request.then(value => {
      if (!Array.isArray(value.data)) throw new Error('Invalid directory list');
      if (!controller.signal.aborted) {
        setDirectoryPath(value.current_path || value.current_name || '');
        setState({ loading: false, value, error: null });
      }
    }).catch(error => { if (!controller.signal.aborted) setState(previous => ({ ...previous, loading: false, error })); });
    return () => controller.abort();
  }, [api, location, refreshKey, open, modelId, startPath]);
  const choose = async (fileName) => {
    const id = state.value?.current_id;
    if (!id || state.loading || mutation.busy || mutation.unknown) return;
    const path = state.value?.current_path;
    const result = await mutation.run(signal => api.register(id, signal, fileName), 'registration');
    if (result) { if (typeof path === 'string') onRegistered?.(path); onOpenChange(false); }
  };
  const catalog = {
    modelBrowser: state.value,
    modelBrowserOpen: open,
    modelBrowserError: state.error?.details?.core_error?.message || state.error?.message || '',
    modelBrowserLoading: state.loading,
    modelFilesMode: Boolean(modelId),
    setModelBrowserOpen: onOpenChange,
    modelBrowserTriggerRef: triggerRef,
    busy: state.loading || mutation.busy || mutation.unknown,
    jumpToModelDirectory: event => {
      event.preventDefault();
      const path = directoryPath.trim();
      if (path && !state.loading && !mutation.busy && !mutation.unknown) setLocation({ path });
    },
    openModelDirectory: id => setLocation({ id: id ?? null }),
    modelDirectoryPath: directoryPath,
    setModelDirectoryPath: setDirectoryPath,
    registerCurrentModelDirectory: () => choose(),
    openCurrentDirectory: async () => {
      const id = state.value?.current_id;
      if (!id || state.loading) return;
      const opened = await mutation.run(signal => api.openDirectory(id, signal));
      if (opened) onOpenChange(false);
    },
  };
  return <ModelDirectoryDialog catalog={catalog} navigation={modelId ? 'id' : 'path'}
    registerModelFile={modelId ? undefined : choose} registering={mutation.busy} />;
}
