import React, { useEffect, useRef, useState } from 'react';
import { ModelDeleteDialog } from '../../../studio/web/src/features/models/ModelDeleteDialog.tsx';
import { LocalCheckpoints } from '../../../studio/web/src/features/models/LocalCheckpoints.tsx';
import { Icon, SectionLabel, TMPanel } from '../../../studio/web/src/app/display.tsx';
import { CoreError } from './page-kit.jsx';
import { ModelLibrarySummary } from '../../../studio/web/src/features/models/ModelLibrarySummary.tsx';
import { ModelManagement } from './model-panel.jsx';
import { ModelRegistration } from './model-registration.jsx';

function ModelLibrary({ state, runtime, tr, openLoad, openUnload, openSettings, openFiles, loadContextForModel, disabled, chooseDirectory, prepareModelDeletion, deletionBusy, refreshKey, folderPath }) {
  const [query, setQuery] = useState('');
  const artifacts = state.inventory.models;
  const loaded = state.inventory.loadedModels;
  const unlisted = loaded.filter(item => !artifacts.some(artifact => artifact.name === item.model));
  const catalog = {
    runtime, artifacts, instances: state.inventory.instances, busy: disabled || deletionBusy, modelFilter: query, setModelFilter: setQuery,
    prepareModelDeletion,
    filteredArtifacts: artifacts.filter(item => [item.name || item.id, item.dflash2_target].some(name => name?.toLowerCase().includes(query.trim().toLowerCase()))),
    connectionRevision: refreshKey, loadDeviceIds: undefined,
    loadContextForModel: name => loadContextForModel(name) || { jev_weights: 'stream' },
    loadArtifact: (name, companionIds) => { const model = artifacts.find(item => item.name === name); if (model) openLoad(model, companionIds); },
    unloadInstance: id => { const model = loaded.find(item => item.instance_id === id); if (model) openUnload(model); },
    chooseModelDirectory: chooseDirectory,
    openModelSettings: name => { const model = artifacts.find(item => item.name === name); if (model) openSettings(model); },
    openModelFiles: openFiles,
  };
  return <>
    <ModelLibrarySummary artifacts={artifacts} instances={state.inventory.instances} folderPath={folderPath}
      onChooseFolder={chooseDirectory} busy={disabled || deletionBusy} tr={tr} />
    <LocalCheckpoints catalog={catalog} companionMode="both" />
    {unlisted.length > 0 && <><SectionLabel title={tr('运行实例', 'Runtime instances')} /><TMPanel className="model-catalog-panel"><div className="model-list">{unlisted.map(model => <div className="model-row" key={model.instance_id || model.id}><span className="model-state active" /><div className="checkpoint-row-details"><strong>{model.model || model.id}</strong><small>{model.state}</small></div><button type="button" disabled={disabled} onClick={() => openUnload(model)}>{tr('卸载', 'Unload')}</button></div>)}</div></TMPanel></>}
  </>;
}
export function ModelsContent({ values, state, actions, t, api, mutation, refreshKey, refresh, tr, directoryOpen, onDirectoryChange, chooseDirectory, directoryTrigger, folderPath, rememberFolder }) {
  const [filesModel, setFilesModel] = useState(null);
  const fileTrigger = useRef(null);
  const deleteTrigger = useRef(null);
  const deletionRequest = useRef(null);
  const [deletion, setDeletion] = useState({ plan: null, loading: false, error: null });
  const runtime = values.status?.error ? null : values.status?.value;
  const catalogState = state.inventory && values.instances?.value
    ? { ...state, inventory: { ...state.inventory, instances: values.instances.value.data } } : state;
  useEffect(() => { deletionRequest.current?.abort(); setDeletion({ plan: null, loading: false, error: null }); }, [refreshKey]);
  useEffect(() => () => deletionRequest.current?.abort(), []);
  async function prepareModelDeletion(id) {
    if (mutation.busy || mutation.unknown || deletion.loading) return;
    deleteTrigger.current = document.activeElement; deletionRequest.current?.abort();
    const controller = new AbortController(); deletionRequest.current = controller;
    setDeletion({ plan: null, loading: true, error: null });
    try {
      const plan = await api.deletionPlan(id, controller.signal);
      if (plan?.model_id !== id || !Array.isArray(plan.files) || !/^[0-9a-f]{64}$/.test(plan.token) || !Number.isFinite(plan.total_bytes)) throw new Error('Invalid model deletion plan');
      if (!controller.signal.aborted) setDeletion({ plan, loading: false, error: null });
    } catch (error) { if (!controller.signal.aborted) setDeletion({ plan: null, loading: false, error }); }
  }
  async function confirmModelDeletion() {
    const plan = deletion.plan; if (!plan || mutation.busy || mutation.unknown) return;
    const result = await mutation.run(signal => api.deleteArtifact(plan.model_id, plan.token, signal));
    if (result) setDeletion({ plan: null, loading: false, error: null });
  }
  return <>{values.status?.error && <CoreError error={values.status.error} />}<ModelRegistration api={api} mutation={mutation} refreshKey={refreshKey} open={directoryOpen} onOpenChange={onDirectoryChange} triggerRef={directoryTrigger} startPath={folderPath} onRegistered={rememberFolder} />
    {catalogState.inventory && <ModelManagement state={catalogState} actions={actions} t={t} tr={tr} api={api} mutation={mutation} refresh={refresh} refreshKey={refreshKey} runtime={runtime}
      draftDownloadAvailable={values.job_kinds?.value?.data?.some(job => job.kind === 'download.huggingface') === true}
      renderCatalog={controls => <ModelLibrary {...controls} folderPath={folderPath} refreshKey={refreshKey} state={catalogState} runtime={runtime} tr={tr} prepareModelDeletion={prepareModelDeletion} deletionBusy={deletion.loading} chooseDirectory={chooseDirectory} openFiles={id => { fileTrigger.current = document.activeElement; setFilesModel(id); }} />} />}
    {deletion.loading && <p role="status">{tr('正在读取删除范围…', 'Reading deletion scope…')}</p>}{deletion.error && <CoreError error={deletion.error} />}
    <ModelDeleteDialog catalog={{ modelDeletion: deletion.plan, busy: mutation.busy || mutation.unknown, deletionTriggerRef: deleteTrigger, cancelModelDeletion: () => setDeletion({ plan: null, loading: false, error: null }), confirmModelDeletion }} />
    {filesModel && <ModelRegistration key={filesModel} api={api} mutation={mutation} refreshKey={refreshKey} open modelId={filesModel} onOpenChange={open => { if (!open) setFilesModel(null); }} triggerRef={fileTrigger} />}
  </>;
}
