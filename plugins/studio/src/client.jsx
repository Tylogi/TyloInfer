import React, { useCallback, useEffect, useId, useRef, useState, useSyncExternalStore } from 'react';
import stylesheet from './studio.css?inline';
import settingsStyles from './settings.css?inline';
import pagesStyles from './pages.css?inline';
import productStyles from './product-shell.css?inline';
import { createModelManager } from './model-manager.js';
import { createOperationCheckpoint } from './operation-checkpoint.js';
import { OperationPanel } from './model-panel.jsx';
import sharedStudioStyles from 'tyloinfer:studio-styles';
import restoredStyles from './restored-pages.css?inline';
import { SharedPresenters } from './shared-presenters.jsx';
import { Icon, ScreenHeader } from '../../../studio/web/src/app/display.tsx';
import { ModelsContent } from './models-page.jsx';
import { modelMessages } from './model-messages.js';
import { createPageApi } from './page-api.js';
import { terminalStates } from './model-api.js';
import { usePageRead, usePageMutation, PageMutationStatus } from './page-kit.jsx';
import { OverviewContent, ResourcesContent, LogsContent } from './runtime-pages.jsx';
import { LogUpdateControl } from '../../../studio/web/src/features/runtime/LogUpdateControl.tsx';
import { ServiceContent } from './service-page.jsx';
import { AnalysisContent } from './analysis-page.jsx';
import { DownloadsContent } from './download-page.jsx';
import { BrandMark, BrandName, HomeMark, ProductTitle, applyProductIcon, productName, productVersion } from './branding.jsx';

const namespace = 'tyloinfer.studio';
const messages = {
  en: { ...modelMessages.en, languageCode: 'en', engine: 'Inference engine', settings: 'Settings', overview: 'Overview', models: 'Models', service: 'Service', resources: 'Resources', downloads: 'Downloads', analysis: 'Analysis', logs: 'Logs', migrating: 'This inference page has not migrated yet', refresh: 'Refresh', loading: 'Connecting to the inference core…', disconnected: 'Inference core unavailable', back: 'Back to chat', loaded: 'Registered models', instances: 'Running instances', empty: 'No model loaded', light: 'Light', dark: 'Dark', system: 'Automatic appearance', unknownVersion: 'unknown version' },
  zh: { ...modelMessages.zh, languageCode: 'zh', engine: '推理引擎', settings: '设置', overview: '概览', models: '模型', service: '服务', resources: '资源', downloads: '下载', analysis: '分析', logs: '日志', migrating: '此推理页面尚未迁移', refresh: '刷新', loading: '正在连接推理核心…', disconnected: '推理核心不可用', back: '返回对话', loaded: '已登记模型', instances: '运行实例', empty: '未加载模型', light: '浅色', dark: '深色', system: '自动明暗', unknownVersion: '版本未知' },
};
const entries = ['overview', 'models', 'service', 'resources', 'downloads', 'analysis', 'logs'];
const mainKey = 'tyloinfer.inference';
const modelFolderKey = 'tyloinfer.agent.model-folder.v1';

const inferenceIcons = { overview: 'gauge', models: 'folder', service: 'server-rack', resources: 'droplet', downloads: 'download', analysis: 'search', logs: 'scroll' };
function InferenceIcon({ name }) {
  return <Icon name={inferenceIcons[name]} size={17} />;
}
function EngineLauncher({ wide, settingsOpen, openSettings, t, showInference, hostVersion, useNavSelection, usePanelInfo }) {
  const [open, setOpen] = useState(true);
  const selected = useNavSelection(value => value);
  const active = usePanelInfo(info => info.activePanelId === mainKey);
  const region = useId();
  return <div className={`tylo-studio-launcher ${wide ? '' : 'tylo-studio-rail'}`}>
    <nav id={region} hidden={!open || !wide} aria-label={t('engine')}>
      {entries.map((entry) => <button type="button" key={entry} aria-current={active && selected === entry ? 'page' : undefined}
        onClick={() => showInference(entry)}><InferenceIcon name={entry} /><span>{t(entry)}</span></button>)}
    </nav>
    <button type="button" className="tylo-studio-engine" aria-expanded={open} aria-controls={region}
      aria-label={t('engine')} onClick={() => { if (!wide) showInference('overview'); else setOpen((value) => !value); }}>
      <Icon name="server-rack" size={18} />{wide && <span>{t('engine')}</span>}
      {wide && <span className="tylo-studio-triangle" data-open={open} aria-hidden="true" />}
    </button>
    {openSettings && <button type="button" className="tylo-studio-settings" aria-haspopup="dialog" aria-expanded={settingsOpen}
      aria-label={t('settings')} onClick={() => openSettings()}><Icon name="settings" size={18} />{wide && <span>{t('settings')}</span>}</button>}
    {wide && <div className="tylo-studio-version"><span>{productName} {productVersion}</span>{' '}<span>based on DSH {hostVersion || t('unknownVersion')}</span></div>}
  </div>;
}

function DSBotEngineLauncher(props) {
  const [placement, setPlacement] = useState(null);
  useEffect(() => {
    const mount = document.createElement('div');
    mount.className = 'tylo-studio-dsbot-footer';
    const place = () => {
      const footer = document.querySelector('.bt-leftcol:not([data-leaving]) .bt-leftcol-foot');
      if (!footer) {
        mount.remove();
        setPlacement(null);
        return;
      }
      if (mount.nextElementSibling !== footer) footer.before(mount);
      const wide = !footer.classList.contains('bt-leftcol-foot-rail');
      setPlacement(current => current?.wide === wide ? current : { mount, wide });
    };
    const observer = new MutationObserver(place);
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-leaving', 'data-left'] });
    place();
    return () => { observer.disconnect(); mount.remove(); };
  }, []);
  if (placement) return createDSBotPortal(<EngineLauncher {...props} wide={placement.wide} />, placement.mount);
  return <div className="tylo-studio-dsbot-classic"><EngineLauncher {...props} /></div>;
}

function InferencePanel({ t, useSelection, useModelState, watchModels, modelActions, pageApi, showInference, openProductSettings, backToChat, theme, subscribeTheme, pageCheckpoint }) {
  const appearance = useSyncExternalStore(subscribeTheme, theme.getTheme.bind(theme), theme.getTheme.bind(theme));
  const view = useSelection(value => value);
  const pageElement = useRef(null);
  useEffect(() => { pageElement.current?.scrollTo({ top: 0 }); }, [view]);
  const state = useModelState(value => value);
  const [refreshKey, setRefreshKey] = useState(0);
  const [directoryOpen, setDirectoryOpen] = useState(false);
  const [logsPaused, setLogsPaused] = useState(false);
  const [analysisSelection, setAnalysisSelection] = useState('');
  const [folderPath, setFolderPath] = useState(() => {
    try { return window.localStorage.getItem(modelFolderKey); } catch { return null; }
  });
  const directoryTrigger = useRef(null);
  const chooseDirectory = event => {
    directoryTrigger.current = event?.currentTarget ?? document.activeElement;
    setDirectoryOpen(true);
  };
  const rememberFolder = path => {
    setFolderPath(path);
    try { window.localStorage.setItem(modelFolderKey, path); } catch {}
  };
  useEffect(() => { setDirectoryOpen(false); }, [view]);
  const [filters, setFilters] = useState({ query: '', sort: 'recent', offset: 0, limit: 100 });
  const language = t('languageCode');
  const tr = useCallback((zh, en) => language === 'zh' ? zh : en, [language]);
  const refresh = useCallback(() => { void modelActions.refresh(); setRefreshKey(value => value + 1); }, [modelActions]);
  const [selectedModel, setSelectedModel] = useState('');
  const selectedInstance = state.inventory?.instances.find(item => item.model === selectedModel && ['ready', 'busy'].includes(item.state)) || state.inventory?.instances.find(item => ['ready', 'busy'].includes(item.state));
  const selectedName = selectedInstance?.model || '';
  const settled = view === 'models' && terminalStates.has(state.operation?.status) ? state.operation.id : '';
  const page = usePageRead(pageApi, view, { prefix_cache: filters, logs: { limit: 500 }, metrics: { limit: 500 },
    jobs: { limit: ['overview', 'models'].includes(view) ? 40 : 200 },
    serving: { scope: 'session', ...(view === 'overview' && selectedName ? { model: selectedName } : {}) },
    ...(['overview', 'resources'].includes(view) && selectedInstance ? { status: { instance_id: selectedInstance.id } } : {}) }, `${refreshKey}:${settled}`);
  const liveInstanceIds = page.values.instances?.value?.data?.map(item => item.id).sort().join(',');
  useEffect(() => {
    if (['overview', 'models'].includes(view) && liveInstanceIds != null && state.inventory &&
        liveInstanceIds !== state.inventory.instances.map(item => item.id).sort().join(',')) void modelActions.refresh();
  }, [view, liveInstanceIds, modelActions]);
  const mutation = usePageMutation(pageApi, refresh, refreshKey, pageCheckpoint, view);
  const inventoryReady = Boolean(state.inventory) && !state.loading && !state.inventoryError;
  const reviewReady = inventoryReady && (view === 'models' || !page.loading && Object.values(page.values).every(entry => !entry.error));
  useEffect(() => watchModels(), [watchModels]);
  const props = { values: page.values, inventory: state.inventory, api: pageApi, mutation, tr, go: showInference, refreshKey, refresh, directoryOpen, onDirectoryChange: setDirectoryOpen, chooseDirectory, directoryTrigger, selectedModel: selectedName, selectModel: setSelectedModel, loading: page.loading, logsPaused, analysisSelection, setAnalysisSelection, folderPath, rememberFolder, inventoryLoading: state.loading, inventoryError: state.inventoryError };
  const subtitles = {
    overview: tr('本机推理状态与性能。', 'Local inference status and performance.'), models: tr('管理本地模型与运行实例。', 'Manage local models and runtime instances.'), service: tr('运行服务、连接与模型默认值。', 'Runtime service, connections, and model defaults.'), resources: tr('监视计算占用、权重与 KV 流量、前缀缓存。', 'Monitor compute utilization, weight and KV traffic, and prefix caching.'), downloads: tr('为设备寻找模型与精度档位。', 'Find models and precision tiers for your device.'), analysis: tr('探索检查点架构、参数与精度。', 'Explore checkpoint architecture, parameters and precision.'), logs: tr('查看服务事件、请求性能与任务记录。', 'Inspect service events, request performance, and job records.'),
  };
  return <section ref={pageElement} className="tylo-studio-inference" aria-label={t('engine')} data-inference-page={view} data-studio-theme={appearance.active?.colorScheme || appearance.preference}>
    <SharedPresenters tr={tr} inventory={state.inventory} runtime={page.values.status?.value} operation={state.operation} jobs={[...(page.values.jobs?.value?.data || []), ...(mutation.operation ? [mutation.operation] : [])]}><div className="tylo-restored-page">
    <ScreenHeader title={view === 'downloads' ? tr('模型下载', 'Model downloads') : t(view)} subtitle={subtitles[view]} trailing={<>{view === 'logs' && <LogUpdateControl paused={logsPaused} onChange={setLogsPaused} tr={tr} />}{view === 'models' && <><button type="button" onClick={() => showInference('downloads')}><Icon name="download" size={14} />{tr('模型下载', 'Model downloads')}</button><button type="button" className="primary" onClick={chooseDirectory}><Icon name="folder" size={14} />{tr('添加模型', 'Add model')}</button></>}<button type="button" onClick={backToChat}>{t('back')}</button><button type="button" disabled={state.loading || page.loading} onClick={refresh}><Icon name="refresh" size={14} />{t('refresh')}</button></>} />
    {(state.loading || page.loading) && <p role="status">{t('loading')}</p>}
    {state.inventoryError && <div role="alert"><p>{t(state.inventory ? 'staleInventory' : 'managementUnavailable')}</p><p>{state.inventoryError.message}</p></div>}
    {view === 'models' && <OperationPanel state={state} actions={modelActions} t={t} />}
    {['models', 'service', 'resources', 'downloads', 'logs'].includes(view) && <PageMutationStatus mutation={mutation} tr={tr} refreshKey={refreshKey} reviewReady={reviewReady} />}
    {view === 'overview' && <OverviewContent {...props} />}
    {view === 'models' && <ModelsContent {...props} state={state} actions={modelActions} t={t} />}
    {view === 'service' && <ServiceContent {...props} openSettings={openProductSettings} />}
    {view === 'resources' && <ResourcesContent {...props} filters={filters} setFilters={setFilters} />}
    {view === 'downloads' && <DownloadsContent {...props} />}
    {view === 'analysis' && <AnalysisContent {...props} />}
    {view === 'logs' && <LogsContent {...props} />}
    </div></SharedPresenters>
  </section>;
}

function ThemeControls({ t, theme, subscribeTheme, setTheme, panelInfo }) {
  const snapshot = useSyncExternalStore(subscribeTheme, theme.getTheme.bind(theme));
  const panel = useSyncExternalStore(panelInfo.subscribe, panelInfo.getSnapshot);
  if (panel.activePanelId !== null) return null;
  return <div className="tylo-studio-theme" role="group" aria-label={t('settings')}>
    {['light', 'dark', 'system'].map((value) => <button type="button" key={value} aria-label={t(value)}
      aria-pressed={snapshot.preference === value} onClick={() => setTheme(value)}>{value === 'light' ? '☀' : value === 'dark' ? '☾' : 'Auto'}</button>)}
  </div>;
}

function applyProductShell(ctx) {
  ctx.effect(() => applyProductIcon(document));
  ctx.slots.inject('sidebar.brand.mark', () => ctx.slots.register({ name: 'sidebar.brand.mark', priority: -100 }, BrandMark));
  ctx.slots.inject('sidebar.brand.name', () => ctx.slots.register({ name: 'sidebar.brand.name', priority: -100 }, BrandName));
  ctx.slots.inject('conversation.hero.brand.mark', () => ctx.slots.register({ name: 'conversation.hero.brand.mark', priority: -100 }, HomeMark));
  ctx.slots.inject('shell.overlay', () => ctx.slots.register({ name: 'shell.overlay', id: 'tyloinfer-title', order: 110 }, ProductTitle));
  ctx.effect(() => {
    document.body.classList.add('tylo-studio-product');
    return () => document.body.classList.remove('tylo-studio-product');
  });
  ctx.effect(() => ctx.theme.overrideTokens('@tyloinfer/dsh-studio', {
    '--dsw-alias-brand-primary': { light: '#9f6232', dark: '#d6aa73' },
    '--dsw-alias-state-business-primary': { light: '#9f6232', dark: '#d6aa73' },
    '--dsw-alias-button-info-fill': { light: '#9f6232', dark: '#895121' },
    '--dsw-alias-button-info-hover': { light: '#895121', dark: '#9f6232' },
    '--dsw-alias-bg-base': { light: '#ffffff', dark: '#222327' },
    '--dsw-specific-sidebar-fill': { light: '#eef0f3', dark: '#1c1d21' },
  }));
}

export const inject = ['slots', 'layout', 'locale', 'theme', 'connection', 'modules'];
export function apply(ctx) {
  ctx.effect(() => ctx.locale.register(namespace, messages));
  if (!__TYLOINFER_DSBOT__) applyProductShell(ctx);
  ctx.effect(() => {
    const style = document.createElement('style');
    style.dataset.tyloinferStudio = '';
    style.textContent = (__TYLOINFER_DSBOT__
      ? [stylesheet, pagesStyles, sharedStudioStyles, restoredStyles]
      : [stylesheet, settingsStyles, pagesStyles, productStyles, sharedStudioStyles, restoredStyles]).join('\n');
    document.head.append(style);
    return () => style.remove();
  });
  const listeners = new Set();
  let view = 'overview';
  const selection = { getSnapshot: () => view, subscribe: (listener) => { listeners.add(listener); return () => listeners.delete(listener); } };
  const showInference = (next) => { view = next; listeners.forEach((listener) => listener()); ctx.layout.selectPanel(mainKey); };
  const modelManager = createModelManager(ctx.connection, { checkpoint: createOperationCheckpoint(() => window.sessionStorage, 'tyloinfer.model-operation.v1') });
  const pageApi = createPageApi(ctx.connection);
  const pageCheckpoint = createOperationCheckpoint(() => window.sessionStorage, 'tyloinfer.page-operation.v1');
  const openProductSettings = __TYLOINFER_DSBOT__ ? undefined : () => document.querySelector('.tylo-studio-settings')?.click();
  const modelActions = { refresh: modelManager.refresh, load: modelManager.load, unload: modelManager.unload, cancel: modelManager.cancel, poll: modelManager.poll, track: modelManager.track, acknowledge: modelManager.acknowledge };
  ctx.effect(() => modelManager.dispose);
  const subscribeTheme = (listener) => ctx.on('theme/change', listener);
  const launcher = __TYLOINFER_DSBOT__
    ? { name: 'sidebar.footer.action', id: mainKey, order: 10 }
    : { name: 'settings.launcher', priority: -100 };
  ctx.slots.inject(launcher.name, () => ctx.slots.register({ ...launcher, locale: namespace,
    inject: () => ({ hooks: { navSelection: selection }, showInference, hostVersion: ctx.modules.manifest.release?.version }) },
  __TYLOINFER_DSBOT__ ? DSBotEngineLauncher : EngineLauncher));
  ctx.slots.inject('main', () => ctx.slots.register({ name: 'main', key: mainKey, locale: namespace,
    inject: () => ({ hooks: { selection, modelState: modelManager }, watchModels: modelManager.watch, modelActions, pageApi, pageCheckpoint, showInference, openProductSettings, theme: ctx.theme, subscribeTheme, backToChat: () => ctx.layout.selectPanel(null) }) }, InferencePanel));
  if (!__TYLOINFER_DSBOT__) {
    ctx.slots.inject('shell.overlay', () => ctx.slots.register({ name: 'shell.overlay', id: 'tyloinfer-theme', order: 100, locale: namespace,
      inject: () => ({ theme: ctx.theme, subscribeTheme, panelInfo: ctx.layout.panelInfo, setTheme: (value) => ctx.theme.setTheme(value) }) }, ThemeControls));
  }
  ctx.effect(() => () => { listeners.clear(); });
}
