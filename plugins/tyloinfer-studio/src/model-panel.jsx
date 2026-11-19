import React, { useEffect, useRef, useState } from 'react';
import { terminalStates } from './model-api.js';
import { Dialog } from './shared-primitives.jsx';
import { LoadedModelSettings } from './loaded-model-settings.jsx';
import { ModelLoadDialog } from './model-load-dialog.jsx';
import { defaultLoadOptions, loadModelDrafts, saveModelDrafts } from './model-load-options.js';

function ErrorDetails({ error }) {
  const core = error?.details?.core_error;
  return <div className="tylo-model-error" role="alert"><p>{error?.message}</p>
    {error?.code && <code>{error.code}</code>}{typeof core?.message === 'string' && <p>{core.message}</p>}
  </div>;
}

export function OperationPanel({ state, actions, t }) {
  const operation = state.operation;
  return <>
    {state.writePending && <p role="status">{t('submitting')}</p>}
    {state.writeError && <ErrorDetails error={state.writeError} />}
    {state.uncertainty && <div className="tylo-model-uncertain" role="alert"><p>{t('unknownOutcome')}</p>
      <button type="button" disabled={state.loading} onClick={() => void actions.refresh()}>{t('refresh')}</button>{' '}
      <button type="button" disabled={!state.uncertainty.reviewed || state.loading} onClick={actions.acknowledge}>{t('acknowledgeUnknown')}</button>
    </div>}
    {operation && <section className="tylo-model-operation" aria-label={t('operation')}>
      <h2>{t(operation.kind === 'model.unload' ? 'unload' : 'load')}{state.operationLabel ? ` · ${state.operationLabel}` : ''}</h2>
      <p role="status">{t(operation.status)}</p>
      {operation.cancel_requested && operation.status !== 'cancelling' && !terminalStates.has(operation.status) && <p>{t('cancelPending')}</p>}
      {Number.isFinite(operation.progress) && <progress aria-label={t('operationProgress')} max={1} value={Math.max(0, Math.min(1, operation.progress))} />}
      {typeof operation.error?.message === 'string' && <p role="alert">{operation.error.message}</p>}
      {['queued', 'running'].includes(operation.status) && <button type="button" disabled={state.writePending || state.cancelIssued || operation.cancel_requested || Boolean(state.uncertainty)} onClick={() => void actions.cancel()}>{t('cancelOperation')}</button>}
      {state.operationError && !terminalStates.has(operation.status) && <button type="button" disabled={state.writePending} onClick={() => void actions.poll()}>{t('refresh')}</button>}
    </section>}
    {state.operationError && <ErrorDetails error={state.operationError} />}
  </>;
}

export function ModelManagement({ state, actions, t, tr, api, mutation, runtime, refresh, refreshKey, renderCatalog }) {
  const [selection, setSelection] = useState(null);
  const [drafts, setDrafts] = useState(() => loadModelDrafts(() => window.sessionStorage));
  const [force, setForce] = useState(false);
  const trigger = useRef(null);
  const mounted = useRef(false);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => { setSelection(current => current?.kind === 'load' ? null : current); }, [refreshKey]);
  const active = state.operation && !terminalStates.has(state.operation.status);
  const mutationLocked = mutation?.busy || mutation?.unknown || Boolean(mutation?.operation && !terminalStates.has(mutation.operation.status));
  const disabled = state.loading || Boolean(state.inventoryError) || state.writePending || state.tracking || Boolean(state.uncertainty) || active || mutationLocked;
  const close = () => setSelection(null);
  const open = (kind, model, companionIds = []) => { trigger.current = document.activeElement; setForce(false); setSelection({ kind, model, companionIds: [...companionIds], loadedInstance: state.inventory.instances.find(item => item.model === model.name && item.state !== 'failed') }); };
  const update = options => {
    const next = { ...drafts, [selection.model.id]: options };
    setDrafts(next); saveModelDrafts(() => window.sessionStorage, next);
  };
  const load = async (request, companionIds) => { if (disabled) return; const accepted = await actions.load(request, companionIds); if (accepted && mounted.current) close(); };
  const unload = async event => { event.preventDefault(); if (disabled || !selection) return; const accepted = await actions.unload({ instance_id: selection.model.instance_id, force }); if (accepted && mounted.current) close(); };
  const controls = { disabled: Boolean(disabled), openLoad: (model, companionIds) => open('load', model, companionIds), openUnload: model => open('unload', model), openSettings: model => open('configure', model), loadContextForModel: name => drafts[state.inventory.models.find(model => model.name === name)?.id]?.context };
  const loaded = selection && state.inventory.instances.find(item => item.model === selection.model.name && ['ready', 'busy', 'unloading'].includes(item.state));
  return <div className="tylo-model-management">
    {renderCatalog(controls)}
    {selection && selection.kind !== 'unload' && !loaded && !selection.loadedInstance && <ModelLoadDialog key={selection.model.id} model={selection.model} companionIds={selection.companionIds} artifacts={state.inventory.models} mode={selection.kind} options={drafts[selection.model.id] || defaultLoadOptions()} onOptions={update} runtime={runtime} api={api} disabled={Boolean(disabled)} t={t} tr={tr} onClose={close} onLoad={load} triggerRef={trigger} />}
    {selection && selection.kind === 'configure' && (loaded || selection.loadedInstance) && <LoadedModelSettings key={selection.model.id} model={selection.model} inventory={state.inventory} runtime={runtime} api={api} mutation={mutation} refresh={refresh} refreshKey={refreshKey} tr={tr} onClose={close} triggerRef={trigger} disabled={state.writePending || active || Boolean(state.uncertainty) || Boolean(state.inventoryError)} />}
    {selection?.kind === 'unload' && <Dialog open title={t('confirmUnload')} closeLabel={t('dismiss')} className="model-settings-dialog" returnFocusRef={trigger} onOpenChange={open => { if (!open && !state.writePending) close(); }}>
      <form className="model-settings-body tylo-model-confirm" aria-label={t('confirmUnload')} onSubmit={unload}><p>{selection.model.model}</p><p>{t('unloadWarning')}</p>
        <label className="tylo-model-check"><input autoFocus type="checkbox" checked={force} disabled={Boolean(disabled)} onChange={event => setForce(event.target.checked)} />{t('forceUnload')}</label>
        <div className="model-settings-actions"><button type="button" disabled={state.writePending} onClick={close}>{t('dismiss')}</button><button type="submit" disabled={Boolean(disabled)}>{t('unload')}</button></div>
      </form>
    </Dialog>}
  </div>;
}
