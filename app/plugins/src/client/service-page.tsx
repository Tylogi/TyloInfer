import React, { useMemo, useState } from 'react';
import { InferencePolicyView } from '@shared/features/connections/InferencePolicyView.tsx';
import { ModelAliasMappingView } from '@shared/features/connections/ModelAliasMappingView.tsx';
import { RemoteRoutingView } from '@shared/features/connections/RemoteRoutingView.tsx';
import { runtimeModelNames } from '@shared/features/runtime/modelSelection.ts';
import { useServiceConfirmation } from './service-confirmation.tsx';
import { CoreError } from './page-kit.tsx';
import { Icon, SectionLabel, SettingRow, TMPanel } from '@shared/ui/display.tsx';
import { PageTabs } from './page-navigation.tsx';
import { MemoryControls } from './memory-controls.tsx';
import { GlobalContext } from './global-context.tsx';
import { AuthenticationControls, ListenerControls } from './service-connection-controls.tsx';

function PolicyPanel({ title, entry, children }) {
  return <><SectionLabel title={title} /><TMPanel className="server-settings-panel"><div className="setting-list">{entry?.error ? <CoreError error={entry.error} /> : entry?.value !== undefined ? children(entry.value) : <p role="status">…</p>}</div></TMPanel></>;
}
export function ServiceContent({ values, inventory, api, mutation, tr, openSettings, refreshKey, selectedModel }) {
  const [section, setSection] = useState('general');
  const [visited, setVisited] = useState(['general']);
  const [remoteError, setRemoteError] = useState('');
  const disabled = mutation.busy || mutation.unknown || Boolean(mutation.operation && !['succeeded', 'failed', 'cancelled', 'interrupted'].includes(mutation.operation.status));
  const confirmation = useServiceConfirmation({ api, mutation, tr, refreshKey, section });
  const save = (resource, request) => void confirmation.configure(resource, request);
  const aliasServices = useMemo(() => ({ modelAliases: () => api.read('model_aliases', {}, confirmation.getSignal()), configureModelAliases: aliases => confirmation.configure('model_aliases', { aliases }) }), [api, confirmation.getSignal, confirmation.configure]);
  const remoteServices = {
    remoteNodes: (refresh = false) => api.remoteNodes(refresh, confirmation.getSignal()),
    createRemoteNode: request => mutation.run(signal => api.addRemoteNode(request, signal)),
    deleteRemoteNode: id => mutation.run(async signal => { await api.deleteRemoteNode(id, signal); return { removed: true }; }),
  };
  const authentication = values.authentication_policy;
  return <div className="service-view"><PageTabs items={[["general", tr('常用', 'General')], ["cache", tr('缓存', 'Cache')], ["advanced", tr('高级', 'Advanced')]]} value={section} onChange={value => { setSection(value); setVisited(current => current.includes(value) ? current : [...current, value]); }} label={tr('服务设置分类', 'Service settings sections')} />
    <div className="server-page" role="tabpanel" aria-label={tr('服务配置', 'Service configuration')} hidden={section !== 'general'}>
      <div className="server-active-notice"><Icon name="info" size={15} /><span>{tr('服务连接由 Host 管理；修改推理策略无需重新打开 Agent，也不会自动加载模型。', 'The Host manages the connection. Updating inference policy does not reopen the Agent or automatically load a model.')}</span></div>
      <PolicyPanel title={tr('API 参数', 'API parameters')} entry={values.listener}>{value => <>
        <SettingRow title={tr('模型 ID', 'Model ID')} detail={tr('用于对话补全请求。', 'Used for chat completions.')} trailing={<div className="server-row-actions server-model-control"><ModelAliasMappingView key={refreshKey} models={runtimeModelNames([], inventory?.instances || [])} selectedModel={selectedModel} services={aliasServices} connectionScope={confirmation.connectionScope} notify={() => {}} writeBlocked={disabled} /></div>} />
        {values.model_aliases?.error && <CoreError error={values.model_aliases.error} />}
        {openSettings && <SettingRow title={tr('服务连接', 'Service connection')} trailing={<button type="button" onClick={openSettings}>{tr('Agent 提供商设置', 'Agent provider settings')}</button>} />}
        <ListenerControls listener={value} api={api} mutation={mutation} disabled={disabled} tr={tr} />
        <AuthenticationControls entry={authentication} api={api} mutation={mutation} disabled={disabled} tr={tr} />
      </>}</PolicyPanel>
      <MemoryControls values={values} inventory={inventory} api={api} mutation={mutation} tr={tr} refreshKey={refreshKey} />
      {values.inference_policy?.error && <CoreError error={values.inference_policy.error} />}
      <InferencePolicyView enabled={values.inference_policy?.value?.mtp_enabled} ready={Boolean(values.inference_policy?.value)} busy={disabled} changeMtp={next => save('inference_policy', { mtp_enabled: next })} tr={tr} />
      <GlobalContext key={refreshKey} values={values} inventory={inventory} confirmation={confirmation} refreshKey={refreshKey} disabled={disabled} tr={tr} api={api} mutation={mutation} />
    </div>
    <div className="server-page" role="tabpanel" aria-label={tr('缓存配置', 'Cache configuration')} hidden={section !== 'cache'}>{visited.includes('cache') && <><MemoryControls cache values={values} inventory={inventory} api={api} mutation={mutation} tr={tr} refreshKey={refreshKey} /><p className="configuration-effect">{tr('前缀清单、分层占用与缓存维护在资源页查看；不会自动清理缓存。', 'Inspect cache inventory, tier usage and maintenance on Resources. No cache is cleared automatically.')}</p></>}</div>
    <div className="server-page" role="tabpanel" aria-label={tr('高级配置', 'Advanced configuration')} hidden={section !== 'advanced'}>{section === 'advanced' && <>
      <RemoteRoutingView key={refreshKey} ready={Boolean(values.listener?.value)} connectionRevision={refreshKey}
        connectionScope={confirmation.connectionScope} services={remoteServices} writeBlocked={disabled} reportError={setRemoteError} autoRefresh={false} />
      {remoteError && <CoreError error={remoteError} floating />}
    </>}</div>
    {confirmation.presentation}
  </div>;
}
