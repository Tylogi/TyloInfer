import React, { useEffect, useRef, useState } from 'react';
import { terminalStates } from './model-api.js';
import { Dialog, Switch } from './shared-primitives.jsx';
import { ModelMonogram } from '../../../studio/web/src/app/display.tsx';
import { LoadedModelSettings } from './loaded-model-settings.jsx';
import { ModelLoadDialog } from './model-load-dialog.jsx';
import { defaultLoadOptions, loadModelDrafts, saveModelDrafts } from './model-load-options.js';
import { CoreError } from './page-kit.jsx';

export function OperationPanel({ state, actions, t }) {
  const operation = state.operation;
  const error = state.writeError ?? state.operationError ?? operation?.error;
  const previous = useRef(error);
  return <>
    {state.writePending && <p role="status">{t('submitting')}</p>}
    {error && error !== previous.current && <CoreError error={error} floating />}
    {state.uncertainty && <div className="tylo-model-uncertain" role="alert"><p>{t('unknownOutcome')}</p>
      <button type="button" disabled={state.loading} onClick={() => void actions.refresh()}>{t('refresh')}</button>{' '}
      <button type="button" disabled={!state.uncertainty.reviewed || state.loading} onClick={actions.acknowledge}>{t('acknowledgeUnknown')}</button>
    </div>}
    {operation && <section className="tylo-model-operation" aria-label={t('operation')}>
      <h2>{t(operation.kind === 'model.unload' ? 'unload' : 'load')}{state.operationLabel ? ` · ${state.operationLabel}` : ''}</h2>
      <p role="status">{t(operation.status)}</p>
      {operation.cancel_requested && operation.status !== 'cancelling' && !terminalStates.has(operation.status) && <p>{t('cancelPending')}</p>}
      {!terminalStates.has(operation.status) && <progress aria-label={t('operationProgress')} max={1}
        value={Number.isFinite(operation.progress) && (operation.kind !== 'model.load' || operation.progress > 0.02)
          ? Math.max(0, Math.min(1, operation.progress)) : undefined} />}
      {['queued', 'running'].includes(operation.status) && <button type="button" disabled={state.writePending || state.cancelIssued || operation.cancel_requested || Boolean(state.uncertainty)} onClick={() => void actions.cancel()}>{t('cancelOperation')}</button>}
      {state.operationError && !terminalStates.has(operation.status) && <button type="button" disabled={state.writePending} onClick={() => void actions.poll()}>{t('refresh')}</button>}
    </section>}
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
  const open = (kind, model, companionIds = []) => { trigger.current = document.activeElement; setForce(false); setSelection({ kind, model, companionIds: [...companionIds], previousError: state.writeError }); };
  const update = options => {
    const next = { ...drafts, [selection.model.id]: options };
    setDrafts(next); saveModelDrafts(() => window.sessionStorage, next);
  };
  const load = async (request, companionIds) => { if (disabled) return; const accepted = await actions.load(request, companionIds); if (accepted && mounted.current) close(); };
  const unload = async event => { event.preventDefault(); if (disabled || !selection) return; const accepted = await actions.unload({ instance_id: selection.model.instance_id, force }); if (accepted && mounted.current) close(); };
  const controls = { disabled: Boolean(disabled), openLoad: (model, companionIds) => open('load', model, companionIds), openUnload: model => open('unload', model), openSettings: model => open('configure', model), loadContextForModel: name => drafts[state.inventory.models.find(model => model.name === name)?.id]?.context };
  const loaded = selection && state.inventory.instances.some(item => item.model === selection.model.name && item.state !== 'failed');
  return <div className="tylo-model-management">
    {renderCatalog(controls)}
    {selection && selection.kind !== 'unload' && !loaded && <ModelLoadDialog key={selection.model.id} model={selection.model} companionIds={selection.companionIds} artifacts={state.inventory.models} mode={selection.kind} options={drafts[selection.model.id] || defaultLoadOptions()} onOptions={update} runtime={runtime} api={api} disabled={Boolean(disabled)} t={t} tr={tr} onClose={close} onLoad={load} triggerRef={trigger} />}
    {selection && selection.kind === 'configure' && loaded && <LoadedModelSettings key={selection.model.id} model={selection.model} inventory={state.inventory} runtime={runtime} api={api} mutation={mutation} refresh={refresh} refreshKey={refreshKey} tr={tr} onClose={close} triggerRef={trigger} disabled={state.writePending || active || Boolean(state.uncertainty) || Boolean(state.inventoryError)} />}
    {selection?.kind === 'unload' && <Dialog open title={t('confirmUnload')} closeLabel={tr('关闭', 'Close')} className="model-unload-dialog" returnFocusRef={trigger} onOpenChange={open => { if (!open && !state.writePending) close(); }}>
      <form aria-label={t('confirmUnload')} onSubmit={unload}>
        <div className="model-unload-content">
          <div className="model-unload-identity">
            <ModelMonogram name={selection.model.model} state="ready" />
            <div><strong>{selection.model.model}</strong><small>{t('unloadWarning')}</small></div>
          </div>
          <div className="model-unload-force">
            <div><strong>{t('forceUnload')}</strong><small>{t('forceUnloadDetail')}</small></div>
            <Switch checked={force} onCheckedChange={setForce} label={t('forceUnload')} disabled={Boolean(disabled)} />
          </div>
          {state.writeError && state.writeError !== selection.previousError && <CoreError error={state.writeError} />}
        </div>
        <footer>
          <button type="button" disabled={state.writePending} onClick={close}>{t('dismiss')}</button>
          <button type="submit" className={force ? 'danger' : 'primary'} disabled={Boolean(disabled)}>{t('unload')}</button>
        </footer>
      </form>
    </Dialog>}
  </div>;
}
