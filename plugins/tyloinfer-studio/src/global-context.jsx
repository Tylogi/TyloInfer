import React, { useMemo, useState } from 'react';
import { ModelContextSettingsView } from '../../../TyloInferStudio/src/features/runtime/ModelContextSettingsView.tsx';
import { SectionLabel, TMPanel } from '../../../TyloInferStudio/src/app/display.tsx';

const profiles = () => ({ cacheProfiles: {}, cacheErrors: {} });
const unavailable = () => Promise.reject(new Error('Global instance reload is not exposed by this bridge'));
export function GlobalContext({ values, inventory, confirmation, refreshKey, disabled, tr, api, mutation }) {
  const [notice, setNotice] = useState('');
  const services = useMemo(() => ({
    runtimeApi: { contextPolicy: () => values.context_policy?.error ? Promise.reject(values.context_policy.error) : Promise.resolve(values.context_policy?.value), configureContextPolicy: request => confirmation.configure('context_policy', request), yarnContextInfo: unavailable,
      async reloadGlobalContextCap() {
        const job = await mutation.run(signal => api.configureContext({ apply_global_cap: true }, signal), 'context');
        if (!job) throw new Error(tr('重载请求未确认，请查看服务页的提示。', 'The reload was not confirmed. Check the service status.'));
        return job;
      } },
    connectionScope: confirmation.connectionScope,
    toast: { success: setNotice, warning: setNotice, error: setNotice }, useModelCacheProfiles: profiles,
  }), [values.context_policy, confirmation.configure, confirmation.connectionScope, api, mutation.run, tr]);
  const tracked = mutation.operation?.kind === 'runtime.context.configure' ? mutation.operation : null;
  const jobs = values.jobs?.value?.data || [];
  const state = { runtime: values.status?.value, instances: inventory?.instances || [], ready: Boolean(values.context_policy), connectionRevision: refreshKey, reloadingInstances: {}, reloadModelContext: unavailable, addJob() {} };
  return <><SectionLabel title={tr('上下文默认值', 'Context defaults')} /><TMPanel className="server-settings-panel server-context-panel"><div className="setting-list">
    <ModelContextSettingsView scope="global" runtimeState={state} jobs={tracked ? [tracked, ...jobs.filter(job => job.id !== tracked.id)] : jobs} services={services} writeBlocked={disabled} />
    {notice && <p role="status">{notice}</p>}
  </div></TMPanel></>;
}
