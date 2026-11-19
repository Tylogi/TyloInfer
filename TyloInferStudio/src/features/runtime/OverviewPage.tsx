import { useEffect, useRef, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import {
  Icon,
  ScreenHeader,
  SectionLabel,
  EmptyPanel,
} from '../../app/display';
import { errorMessage } from '../../app/formatters';
import { runtimeModelNames } from './modelSelection';
import { ServingStatisticsPanel, ServingTotalsPanel } from './ServingStatisticsPanel';
import { useServingStatistics } from './useServingStatistics';
import { RuntimeHero } from './RuntimeHero';
import { MemoryHierarchy } from './MemoryHierarchy';
import { residentMemoryCapacity } from './memoryArchitecture';
import { anthropicEndpoint, jevEndpoint, openAIEndpoint } from './endpoint';
import { getApiBaseUrl, getApiToken } from '../../shared/api/client';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { toast } from '../../stores/toastStore';
import { OverviewModels, OverviewEndpoints } from './OverviewPanels';
import { AppearanceControls } from '../../app/AppearanceControls';

export function OverviewPage() {
  const {
    runtime,
    models,
    instances,
    studio,
    selectedModel: model,
    setSelectedModel: selectModel,
    refreshRuntime,
    loading: busy,
    connectionRevision,
  } = useRuntime();
  const { tr } = useSettings();
  const [scope, setScope] = useState<'session' | 'alltime'>('session');
  const { data: servingStatistics, error: statisticsError } = useServingStatistics(scope, connectionRevision);
  const [endpointCopied, setEndpointCopied] = useState<string | null>(null);
  const [listener, setListener] = useState<{ url: string; revision: number; port: number | null } | null>(null);
  const serviceUrl = studio?.service_url || getApiBaseUrl() || window.location.origin;
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  useEffect(() => {
    let disposed = false;
    let pending = false;
    const endpoint = getApiBaseUrl();
    const credential = getApiToken();
    const current = () => !disposed && endpoint === getApiBaseUrl() && credential === getApiToken();
    const refresh = async () => {
      if (pending) return;
      pending = true;
      try {
        const value = await runtimeApi.runtimeListener();
        if (current()) setListener({ url: serviceUrl, revision: connectionRevision, port: value.anthropic_port ?? null });
      } catch {
        if (current()) setListener(null);
      } finally {
        pending = false;
      }
    };
    void refresh();
    const poll = setInterval(() => void refresh(), 5000);
    window.addEventListener('focus', refresh);
    return () => { disposed = true; clearInterval(poll); window.removeEventListener('focus', refresh); };
  }, [serviceUrl, connectionRevision]);
  const availableModelNames = runtimeModelNames(models, instances);
  const anthropicPort = listener?.url === serviceUrl && listener.revision === connectionRevision ? listener.port : null;
  const endpoints = [
    { id: 'openai', name: 'OpenAI', label: tr('OpenAI 兼容端点', 'OpenAI-compatible endpoint'), url: openAIEndpoint(serviceUrl) },
    { id: 'anthropic', name: 'Anthropic', label: tr('Anthropic 兼容端点', 'Anthropic-compatible endpoint'), url: anthropicEndpoint(serviceUrl, anthropicPort) },
    { id: 'jev', name: 'Jev', label: tr('Jev 决策端点', 'Jev decision endpoint'), url: jevEndpoint(serviceUrl) },
  ];
  async function copyEndpoint(id: string, endpoint: string) {
    try {
      await navigator.clipboard.writeText(endpoint);
      setEndpointCopied(id);
      toast.success(tr('服务地址已复制到剪贴板', 'Endpoint URL copied to clipboard'));
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => setEndpointCopied(null), 1600);
    } catch (cause) {
      toast.error(errorMessage(cause));
    }
  }

  return (
    <section className="dashboard-view" id="dashboard-overview">
      <ScreenHeader
        title={tr('概览', 'Overview')}
        subtitle={tr('本机推理状态与性能。', 'Local inference status and performance.')}
        trailing={
          <>
          <button disabled={busy} onClick={() => void refreshRuntime()} type="button">
            <Icon name="refresh" size={14} />
            {tr('刷新', 'Refresh')}
          </button>
          <AppearanceControls />
          </>
        }
      />
      <RuntimeHero />
      <OverviewModels names={availableModelNames} instances={instances} runtime={runtime} selectedModel={model} busy={busy} selectModel={selectModel} tr={tr} />
      <ServingStatisticsPanel data={servingStatistics} error={statisticsError} scope={scope} onScopeChange={setScope} />
      <SectionLabel title={tr('资源概览', 'Resource overview')} />
      {runtime ? (
        <MemoryHierarchy instances={instances} runtime={runtime} connectionRevision={connectionRevision}
          memoryCapacityBytes={residentMemoryCapacity(runtime)} />
      ) : (
        <EmptyPanel
          icon="memory"
          title={tr('推理内存尚未上报', 'Runtime memory is unavailable')}
          message={tr(
            '服务器就绪后会显示模型驻留与缓存状态。',
            'Memory residency and cache state appear when the server is ready.',
          )}
        />
      )}
      <div className="overview-footer-grid">
        <OverviewEndpoints endpoints={endpoints} copied={endpointCopied} copyEndpoint={(id, url) => void copyEndpoint(id, url)} tr={tr} />
        <ServingTotalsPanel data={servingStatistics} scope={scope} runtime={runtime} />
      </div>
    </section>
  );
}
