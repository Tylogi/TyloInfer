import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ConfirmationDialog } from './shared-primitives.jsx';
import { Facts } from './page-kit.jsx';
import { terminalStates } from './model-api.js';

export function useServiceConfirmation({ api, mutation, tr, refreshKey, section }) {
  const [target, setTarget] = useState(null);
  const pending = useRef(null);
  const controller = useRef(null);
  const live = useRef(mutation); live.current = mutation;
  const getSignal = useCallback(() => { if (!controller.current || controller.current.signal.aborted) controller.current = new AbortController(); return controller.current.signal; }, []);
  const connectionScope = useCallback(() => { const signal = getSignal(); return () => !signal.aborted; }, [getSignal]);
  const cancel = useCallback(() => { const value = pending.current; pending.current = null; setTarget(null); value?.resolve(null); }, []);
  useEffect(() => { cancel(); return () => { controller.current?.abort(); pending.current?.resolve(null); pending.current = null; }; }, [refreshKey, section, cancel]);
  const configure = useCallback((resource, request) => new Promise(resolve => {
    if (pending.current || live.current.busy || live.current.unknown) { resolve(null); return; }
    const value = { resource, request: structuredClone(request), resolve }; pending.current = value; setTarget(value);
  }), []);
  const blocked = mutation.busy || mutation.unknown || Boolean(mutation.operation && !terminalStates.has(mutation.operation.status));
  async function confirm() {
    const value = pending.current; if (!value || value !== target || blocked) return;
    pending.current = null; setTarget(null);
    value.resolve(await live.current.run(signal => api.configure(value.resource, value.request, signal)));
  }
  const presentation = <ConfirmationDialog open={Boolean(target)} tr={tr} title={tr('应用服务策略', 'Apply service policy')}
    description={tr('立即更新以下服务策略，运行中的模型无需重载。', 'Apply the following service policy immediately. Running models stay loaded.')}
    confirmLabel={tr('应用', 'Apply')} onCancel={cancel} onConfirm={confirm} disabled={blocked}>
    <Facts entries={Object.entries(target?.request || {}).flatMap(([key, value]) => key === 'aliases' ? Object.entries(value).map(([name, alias]) => [name, alias || tr('原模型名', 'Original model name')]) : [[key === 'mtp_enabled' ? 'MTP' : key === 'max_context_size' ? tr('全局上下文上限', 'Global context limit') : key, value === null ? tr('自动', 'Automatic') : typeof value === 'boolean' ? value ? tr('启用', 'Enabled') : tr('关闭', 'Disabled') : String(value)]])} />
  </ConfirmationDialog>;
  return { configure, getSignal, connectionScope, presentation };
}
