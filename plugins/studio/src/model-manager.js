import { createModelApi, isOperationId, terminalStates } from './model-api.js';
import { describeCoreRuntime } from './core-runtime.js';
import { companionsAvailable, validCompanionPaths } from './companion-options.js';

export function createModelManager(connection, { checkpoint, schedule = callback => { const timer = setTimeout(callback, 1000); return () => clearTimeout(timer); } } = {}) {
  const api = createModelApi(connection);
  const listeners = new Set();
  const lifetime = new AbortController();
  const saved = checkpoint?.load();
  const restored = saved && ['model.load', 'model.unload'].includes(saved.kind) && isOperationId(saved.id);
  let state = { inventory: null, loading: false, inventoryError: null, writePending: false, tracking: Boolean(restored), operation: restored ? { id: saved.id, kind: saved.kind, status: 'accepted', cancel_requested: false } : null, operationLabel: saved?.label || '', operationError: null, writeError: null, uncertainty: saved?.unknown ? { reviewed: false } : null, cancelIssued: false };
  let watching = 0;
  let readController;
  let operationController;
  let cancelTimer;
  const publish = patch => {
    if (lifetime.signal.aborted) return;
    state = { ...state, ...patch };
    const pending = state.operation && !terminalStates.has(state.operation.status);
    const unknown = Boolean(state.uncertainty || (state.writePending && !state.operation));
    checkpoint?.save(pending || unknown ? { id: state.operation?.id, kind: state.operation?.kind, label: state.operationLabel, unknown } : null);
    for (const listener of listeners) listener();
  };
  const stopTimer = () => { cancelTimer?.(); cancelTimer = undefined; };
  const active = () => state.operation && !terminalStates.has(state.operation.status);
  const writable = () => !lifetime.signal.aborted && state.inventory && !state.loading && !state.inventoryError && !state.writePending && !state.tracking && !state.uncertainty && !active();
  const refresh = async () => {
    if (lifetime.signal.aborted) return;
    readController?.abort();
    const controller = new AbortController();
    readController = controller;
    const uncertainty = state.uncertainty;
    publish({ loading: true, inventoryError: null });
    try {
      const inventory = await describeCoreRuntime(connection, controller.signal);
      if (!controller.signal.aborted) publish({ inventory, uncertainty: uncertainty && state.uncertainty === uncertainty ? { ...uncertainty, reviewed: true } : state.uncertainty });
    } catch (error) {
      if (!controller.signal.aborted) publish({ inventoryError: error });
    } finally {
      if (!controller.signal.aborted) publish({ loading: false });
    }
  };
  const poll = async () => {
    stopTimer();
    const id = state.operation?.id;
    if (!id || lifetime.signal.aborted || !watching || state.writePending) return;
    operationController?.abort();
    const controller = new AbortController();
    operationController = controller;
    try {
      const operation = await api.get(id, controller.signal);
      if (controller.signal.aborted || lifetime.signal.aborted) return;
      publish({ operation, operationError: null, tracking: false });
      if (terminalStates.has(operation.status)) await refresh();
      else if (watching) cancelTimer = schedule(() => { void poll(); });
    } catch (error) {
      if (!controller.signal.aborted) publish({ operationError: error, tracking: false });
    }
  };
  const failWrite = error => publish({ writeError: error,
    ...(error.code === 'tyloinfer/outcome-unknown' ? { uncertainty: { reviewed: false } } : {}) });
  const submit = async (kind, request, companionIds) => {
    if (!writable()) return false;
    if (kind === 'load') {
      const model = state.inventory.models.find(row => row.id === request.model && row.loadable === true);
      const validCompanions = companionIds === undefined || validCompanionPaths(companionIds)
        || companionsAvailable(state.inventory.models, request.model, companionIds);
      if (!model || Object.hasOwn(request, 'gguf_companions') || !validCompanions || model.gguf_companion_required && !companionIds?.length) return false;
    }
    if (kind === 'unload' && !state.inventory.loadedModels.some(row => row.instance_id === request.instance_id && isOperationId(row.instance_id))) return false;
    publish({ writePending: true, operation: null, cancelIssued: false, writeError: null, operationError: null, operationLabel: kind === 'load' ? state.inventory.models.find(row => row.id === request.model)?.name || '' : state.inventory.loadedModels.find(row => row.instance_id === request.instance_id)?.model || '' });
    try {
      const ticket = await api[kind](request, lifetime.signal, kind === 'load' ? companionIds : undefined);
      publish({ operation: { id: ticket.operation_id, kind: 'model.' + kind, status: 'accepted', cancel_requested: false }, cancelIssued: false });
      return true;
    } catch (error) { failWrite(error); return false; }
    finally { publish({ writePending: false }); if (watching && active() && !state.uncertainty) void poll(); }
  };
  const cancel = async () => {
    const operation = state.operation;
    if (!operation || !['queued', 'running'].includes(operation.status) || operation.cancel_requested || state.cancelIssued || state.writePending || state.uncertainty || lifetime.signal.aborted) return false;
    stopTimer();
    operationController?.abort();
    publish({ writePending: true, cancelIssued: true, writeError: null });
    try {
      const next = await api.cancel(operation.id, lifetime.signal);
      publish({ operation: next });
      if (terminalStates.has(next.status)) await refresh();
      return true;
    } catch (error) { failWrite(error); return false; }
    finally { publish({ writePending: false }); if (watching && active()) void poll(); }
  };
  const track = async id => {
    if (!isOperationId(id) || state.writePending || state.tracking || active() || lifetime.signal.aborted) return false;
    operationController?.abort();
    const controller = new AbortController();
    operationController = controller;
    publish({ operationError: null, tracking: true });
    try {
      const operation = await api.get(id, controller.signal);
      if (controller.signal.aborted || lifetime.signal.aborted) return false;
      publish({ operation, cancelIssued: operation.cancel_requested });
      if (terminalStates.has(operation.status)) await refresh();
      else if (watching) cancelTimer = schedule(() => { void poll(); });
      return true;
    } catch (error) { if (!controller.signal.aborted) publish({ operationError: error, tracking: false }); return false; }
    finally { publish({ tracking: false }); }
  };
  return {
    getSnapshot: () => state,
    subscribe: listener => { listeners.add(listener); return () => listeners.delete(listener); },
    refresh,
    load: (request, companionIds) => submit('load', request, companionIds),
    unload: request => submit('unload', { instance_id: request.instance_id, force: request.force === true }),
    cancel,
    poll,
    track,
    acknowledge: () => { if (state.uncertainty?.reviewed) publish({ uncertainty: null, writeError: null }); },
    watch: () => {
      watching++;
      void refresh();
      if (active()) void poll();
      return () => {
        watching--;
        if (watching === 0) { stopTimer(); readController?.abort(); operationController?.abort(); publish({ loading: false }); }
      };
    },
    dispose: () => {
      lifetime.abort(); stopTimer(); readController?.abort(); operationController?.abort(); listeners.clear();
    },
  };
}
