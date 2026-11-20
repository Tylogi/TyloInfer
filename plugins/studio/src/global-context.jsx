import React, { useMemo, useState } from 'react';
import { ModelContextSettingsView } from '../../../studio/web/src/features/runtime/ModelContextSettingsView.tsx';
import { SectionLabel, TMPanel } from '../../../studio/web/src/app/display.tsx';

const profiles = () => ({ cacheProfiles: {}, cacheErrors: {} });
const unavailable = () => Promise.reject(new Error('Global instance reload is not exposed by this bridge'));
export function GlobalContext({ values, inventory, confirmation, refreshKey, disabled, tr }) {
  const [notice, setNotice] = useState('');
  const services = useMemo(() => ({
    runtimeApi: { contextPolicy: () => values.context_policy?.error ? Promise.reject(values.context_policy.error) : Promise.resolve(values.context_policy?.value), configureContextPolicy: request => confirmation.configure('context_policy', request), yarnContextInfo: unavailable, reloadGlobalContextCap: unavailable },
    connectionScope: confirmation.connectionScope,
    toast: { success: setNotice, warning: setNotice, error: setNotice }, useModelCacheProfiles: profiles,
  }), [values.context_policy, confirmation.configure, confirmation.connectionScope]);
  const state = { runtime: values.status?.value, instances: inventory?.instances || [], ready: Boolean(values.context_policy), connectionRevision: refreshKey, reloadingInstances: {}, reloadModelContext: unavailable, addJob() {} };
  return <><SectionLabel title={tr('上下文默认值', 'Context defaults')} /><TMPanel className="server-settings-panel server-context-panel"><div className="setting-list">
    <ModelContextSettingsView scope="global" runtimeState={state} jobs={values.jobs?.value?.data || []} services={services} writeBlocked={disabled} allowGlobalReload={false} />
    <p className="configuration-effect">{tr('这里只保存下次加载的全局上限；批量重载未开放。请在模型页分别确认运行实例的上下文设置。', 'Only the global cap for future loads is saved here. Batch reload is not exposed. Confirm context changes for individual instances on Models.')}</p>
    {notice && <p role="status">{notice}</p>}
  </div></TMPanel></>;
}
