import React, { useMemo, useState } from 'react';
import { InferencePolicyView } from '../../../studio/web/src/features/connections/InferencePolicyView.tsx';
import { ModelAliasMappingView } from '../../../studio/web/src/features/connections/ModelAliasMappingView.tsx';
import { runtimeModelNames } from '../../../studio/web/src/features/runtime/modelSelection.ts';
import { useServiceConfirmation } from './service-confirmation.jsx';
import { CoreError } from './page-kit.jsx';
import { EmptyPanel, Icon, SectionLabel, SettingRow, TMPanel } from '../../../studio/web/src/app/display.tsx';
import { PageTabs } from './page-navigation.jsx';
import { MemoryControls } from './memory-controls.jsx';
import { GlobalContext } from './global-context.jsx';

function PolicyPanel({ title, entry, children }) {
  return <><SectionLabel title={title} /><TMPanel className="server-settings-panel"><div className="setting-list">{entry?.error ? <CoreError error={entry.error} /> : entry?.value !== undefined ? children(entry.value) : <p role="status">…</p>}</div></TMPanel></>;
}
export function ServiceContent({ values, inventory, api, mutation, tr, openSettings, refreshKey, selectedModel }) {
  const [section, setSection] = useState('general');
  const [visited, setVisited] = useState(['general']);
  const disabled = mutation.busy || mutation.unknown || Boolean(mutation.operation && !['succeeded', 'failed', 'cancelled', 'interrupted'].includes(mutation.operation.status));
  const confirmation = useServiceConfirmation({ api, mutation, tr, refreshKey, section });
  const save = (resource, request) => void confirmation.configure(resource, request);
  const aliasServices = useMemo(() => ({ modelAliases: () => api.read('model_aliases', {}, confirmation.getSignal()), configureModelAliases: aliases => confirmation.configure('model_aliases', { aliases }) }), [api, confirmation.getSignal, confirmation.configure]);
  const authentication = values.authentication_policy;
  return <div className="service-view"><PageTabs items={[["general", tr('常用', 'General')], ["cache", tr('缓存', 'Cache')], ["advanced", tr('高级', 'Advanced')]]} value={section} onChange={value => { setSection(value); setVisited(current => current.includes(value) ? current : [...current, value]); }} label={tr('服务设置分类', 'Service settings sections')} />
    <div className="server-page" role="tabpanel" aria-label={tr('服务配置', 'Service configuration')} hidden={section !== 'general'}>
      <div className="server-active-notice"><Icon name="info" size={15} /><span>{tr('服务连接由 Host 管理；修改推理策略无需重新打开 Agent，也不会自动加载模型。', 'The Host manages the connection. Updating inference policy does not reopen the Agent or automatically load a model.')}</span></div>
      <PolicyPanel title={tr('API 参数', 'API parameters')} entry={values.listener}>{value => <>
        <SettingRow title={tr('模型 ID', 'Model ID')} detail={tr('由 /v1/models 公布，并用于对话补全请求。', 'Advertised by /v1/models and accepted by chat completions.')} trailing={<div className="server-row-actions server-model-control"><ModelAliasMappingView key={refreshKey} models={runtimeModelNames(inventory?.loadedModels || [], inventory?.instances || [])} selectedModel={selectedModel} services={aliasServices} connectionScope={confirmation.connectionScope} notify={() => {}} writeBlocked={disabled} externalError={mutation.error ? <CoreError error={mutation.error} /> : null} /></div>} />
        {values.model_aliases?.error && <CoreError error={values.model_aliases.error} />}
        <SettingRow title={tr('服务连接', 'Service connection')} detail={tr('核心连接由 Host 管理。下方入口是 Agent 提供商设置，不等同于修改核心监听或认证。', 'The Host manages the core connection. Agent provider settings do not modify core listeners or authentication.')} trailing={<button type="button" onClick={openSettings}>{tr('Agent 提供商设置', 'Agent provider settings')}</button>} />
        <SettingRow title={tr('OpenAI 端口', 'OpenAI port')} detail={tr('仅显示核心实际监听；此桥不开放端口修改。', 'Actual core listener only; changing ports is not exposed by this bridge.')} trailing={<strong>{value.port}</strong>} />
        <SettingRow title={tr('Anthropic 端口', 'Anthropic port')} detail={tr('未启用时不伪造可用端点。', 'No endpoint is fabricated when disabled.')} trailing={<strong>{value.anthropic_port ?? tr('未启用', 'Not enabled')}</strong>} />
        <SettingRow title={tr('认证', 'Authentication')} detail={tr('只读取核心认证状态，不读取凭据或修改认证。', 'Reads core authentication status without reading credentials or changing authentication.')} trailing={authentication?.error ? <CoreError error={authentication.error} /> : <strong>{authentication?.value ? authentication.value.enabled ? tr('已启用', 'Enabled') : tr('未启用', 'Not enabled') : '—'}</strong>} />
      </>}</PolicyPanel>
      <MemoryControls values={values} inventory={inventory} api={api} mutation={mutation} tr={tr} refreshKey={refreshKey} />
      {values.inference_policy?.error && <CoreError error={values.inference_policy.error} />}
      <InferencePolicyView enabled={values.inference_policy?.value?.mtp_enabled} ready={Boolean(values.inference_policy?.value)} busy={disabled} changeMtp={next => save('inference_policy', { mtp_enabled: next })} tr={tr} />
      <GlobalContext key={refreshKey} values={values} inventory={inventory} confirmation={confirmation} refreshKey={refreshKey} disabled={disabled} tr={tr} />
    </div>
    <div className="server-page" role="tabpanel" aria-label={tr('缓存配置', 'Cache configuration')} hidden={section !== 'cache'}>{visited.includes('cache') && <><MemoryControls cache values={values} inventory={inventory} api={api} mutation={mutation} tr={tr} refreshKey={refreshKey} /><p className="configuration-effect">{tr('前缀清单、分层占用与缓存维护在资源页查看；不会自动清理缓存。', 'Inspect cache inventory, tier usage and maintenance on Resources. No cache is cleared automatically.')}</p></>}</div>
    <div className="server-page" role="tabpanel" aria-label={tr('高级配置', 'Advanced configuration')} hidden={section !== 'advanced'}>{visited.includes('advanced') && <EmptyPanel icon="server-rack" title={tr('远程路由未开放', 'Remote routing is not exposed')} message={tr('受限桥尚未开放原版远程路由及监听／认证写入；Agent 提供商设置不能代替这些功能。', 'Original remote routing and listener/authentication writes are not exposed by this bridge. Agent provider settings are not equivalent replacements.')} />}</div>
    {confirmation.presentation}
  </div>;
}
