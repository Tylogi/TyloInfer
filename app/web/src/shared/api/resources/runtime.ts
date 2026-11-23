
import type {
  RuntimeCapabilities,
  RuntimeStatus,
  RuntimeResources,
  RuntimeListener,
  RuntimeModel,
  RuntimeInstance,
  RuntimeMetricSnapshot,
  RuntimeLogEntry,
  RealtimeCapabilities,
  VoiceOutputComponentStatus,
  JobResource,
  ServingStatistics,
} from '@shared/types/api';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

import type {
  RuntimeMemoryPolicy,
  RuntimeAuthenticationPolicy,
  RuntimeContextPolicy,
  KvMemorySettings,
  KvQuantizationSettings,
  QsaKvStorageSettings,
  YarnContextInfo,
  QsaKvOffloadInfo,
  PrefixCacheGroup,
  PrefixCacheBlock,
  PrefixCacheInventory,
  PrefixCacheFilters,
  PrefixCacheText,
} from '@shared/types/runtime';

export type {
  RuntimeMemoryPolicy,
  RuntimeAuthenticationPolicy,
  RuntimeContextPolicy,
  KvMemorySettings,
  KvQuantizationSettings,
  QsaKvStorageSettings,
  YarnContextInfo,
  QsaKvOffloadInfo,
  PrefixCacheGroup,
  PrefixCacheBlock,
  PrefixCacheInventory,
  PrefixCacheFilters,
  PrefixCacheText,
};

export const runtimeApi = {
  servingStatistics(scope: 'session' | 'alltime', signal?: AbortSignal): Promise<ServingStatistics> {
    return request(`/api/v1/runtime/serving?scope=${scope}`, { signal });
  },
  authenticationPolicy(): Promise<RuntimeAuthenticationPolicy> {
    return request('/api/v1/auth/policy');
  },
  configureAuthenticationPolicy(enabled: boolean, apiKey?: string): Promise<RuntimeAuthenticationPolicy> {
    return request('/api/v1/auth/policy', { method: 'PUT', body: JSON.stringify({ enabled, ...(apiKey === undefined ? {} : { api_key: apiKey }) }) });
  },
  qsaKvOffloadInfo(instanceId: string, targetContext: number, signal?: AbortSignal): Promise<QsaKvOffloadInfo> {
    return request(`/api/v1/runtime/qsa-kv/${encodeURIComponent(instanceId)}?target_context=${targetContext}`, { signal });
  },
  configureQsaKvOffload(instanceId: string, policy: Pick<QsaKvOffloadInfo['policy'], 'enabled' | 'budget_bytes'>): Promise<{ operation_id: string }> {
    return request('/api/v1/runtime/qsa-kv', { method: 'PUT', body: JSON.stringify({ instance_id: instanceId, ...policy }) });
  },
  inferencePolicy(): Promise<{ mtp_enabled: boolean }> {
    return request('/api/v1/runtime/inference-policy');
  },
  configureInferencePolicy(mtpEnabled: boolean): Promise<{ mtp_enabled: boolean }> {
    return request('/api/v1/runtime/inference-policy', {
      method: 'PUT', body: JSON.stringify({ mtp_enabled: mtpEnabled }),
    });
  },
  prefixCacheEntries(namespace?: string, offset = 0, signal?: AbortSignal, filters?: PrefixCacheFilters): Promise<PrefixCacheInventory> {
    const query = new URLSearchParams({ offset: String(offset), limit: '100' });
    if (namespace) query.set('namespace', namespace);
    if (filters) Object.entries(filters).forEach(([key, value]) => query.set(key, value));
    return request(`/api/v1/runtime/cache/entries?${query}`, { signal });
  },
  prefixCacheText(namespace: string, block: string, offset = 0): Promise<PrefixCacheText> {
    return request(`/api/v1/runtime/cache/entries/${encodeURIComponent(namespace)}/${encodeURIComponent(block)}/text?offset=${offset}`);
  },
  purgePrefixCache(namespace?: string): Promise<{ released_bytes: number; removed_blocks: number; failed_blocks: number }> {
    return request('/api/v1/runtime/cache/purge', { method: 'POST', body: JSON.stringify({ namespace: namespace ?? null }) });
  },
  runtimeResources(signal?: AbortSignal): Promise<RuntimeResources> {
    return request('/api/v1/runtime/resources', { signal });
  },
  modelAliases(): Promise<{ aliases: Record<string, string> }> {
    return request('/api/v1/runtime/model-aliases');
  },
  configureModelAliases(aliases: Record<string, string>): Promise<{ aliases: Record<string, string> }> {
    return request('/api/v1/runtime/model-aliases', { method: 'PUT', body: JSON.stringify({ aliases }) });
  },
  memoryPolicy(): Promise<RuntimeMemoryPolicy> {
    return request('/api/v1/runtime/memory-policy');
  },
  configureMemoryPolicy(policy: Partial<Pick<RuntimeMemoryPolicy, 'total_limit_bytes' | 'model_limit_bytes' | 'prefix_limit_bytes' | 'prefix_disk_limit_bytes' | 'prefix_ram_limit_bytes' | 'prefix_directory'>>): Promise<{ operation_id: string }> {
    return request('/api/v1/runtime/memory-policy', { method: 'PUT', body: JSON.stringify(policy) });
  },
  runtimeListener(): Promise<RuntimeListener> {
    return request('/api/v1/runtime/listener');
  },

  configureRuntimeListener(port: number, protocol: 'openai' | 'anthropic' = 'openai'): Promise<RuntimeListener> {
    return request('/api/v1/runtime/listener', {
      method: 'PUT', body: JSON.stringify(protocol === 'openai' ? { port } : { port, protocol }),
    });
  },

  runtimeCapabilities(instanceId?: string | null): Promise<RuntimeCapabilities> {
    const suffix = instanceId ? `?instance_id=${encodeURIComponent(instanceId)}` : '';
    return request(`/api/v1/runtime/capabilities${suffix}`);
  },


  runtimeStatus(instanceId?: string | null): Promise<RuntimeStatus> {
    const suffix = instanceId ? `?instance_id=${encodeURIComponent(instanceId)}` : '';
    return request(`/api/v1/runtime/status${suffix}`);
  },


  async runtimeModels(): Promise<RuntimeModel[]> {
    return (await request<{ data: RuntimeModel[] }>('/api/v1/runtime/models')).data;
  },


  async runtimeInstances(): Promise<RuntimeInstance[]> {
    return (await request<{ data: RuntimeInstance[] }>('/api/v1/runtime/instances')).data;
  },

  configureInstance(instanceId: string, policy: { pinned?: boolean; idle_ttl_seconds?: number | null; dflash2_enabled?: boolean }): Promise<RuntimeInstance> {
    return request(`/api/v1/runtime/instances/${encodeURIComponent(instanceId)}`, { method: 'PATCH', body: JSON.stringify(policy) });
  },



  async runtimeMetrics(limit = 200, signal?: AbortSignal, since?: string): Promise<RuntimeMetricSnapshot[]> {
    return (
      await request<{ data: RuntimeMetricSnapshot[] }>(`/api/v1/runtime/metrics?limit=${limit}${since ? `&since=${encodeURIComponent(since)}` : ''}`, { signal })
    ).data;
  },


  async runtimeLogs(limit = 100, signal?: AbortSignal, after?: number): Promise<RuntimeLogEntry[]> {
    return (await request<{ data: RuntimeLogEntry[] }>(`/api/v1/runtime/logs?limit=${limit}${after != null ? `&after=${after}` : ''}`, { signal })).data;
  },


  realtimeCapabilities(): Promise<RealtimeCapabilities> {
    return request('/api/v1/runtime/realtime/capabilities');
  },


  voiceOutputComponent(): Promise<VoiceOutputComponentStatus> {
    return request('/api/v1/components/voice-output');
  },


  installVoiceOutputComponent(): Promise<{ operation_id: string; status: 'accepted' }> {
    return request('/api/v1/components/voice-output/install', { method: 'POST' });
  },


  activateVoiceOutputComponent(): Promise<{ active: boolean; reason?: string; error?: string }> {
    return request('/api/v1/components/voice-output/activate', { method: 'POST' });
  },


  reloadRuntime(contextSize: number, instanceId?: string): Promise<RuntimeStatus> {
    return request('/api/v1/runtime/reload', {
      method: 'POST',
      body: JSON.stringify({ context_size: contextSize, instance_id: instanceId }),
    });
  },

  contextPolicy(): Promise<RuntimeContextPolicy> {
    return request('/api/v1/runtime/context-policy');
  },

  configureContextPolicy(policy: Pick<RuntimeContextPolicy, 'max_context_size'>): Promise<RuntimeContextPolicy> {
    return request('/api/v1/runtime/context-policy', { method: 'PUT', body: JSON.stringify(policy) });
  },

  reloadGlobalContextCap(): Promise<JobResource> {
    return request('/api/v1/runtime/context', { method: 'POST', body: JSON.stringify({ apply_global_cap: true }) });
  },

  yarnContextInfo(instanceId: string): Promise<YarnContextInfo> {
    return request(`/api/v1/runtime/yarn/${encodeURIComponent(instanceId)}`);
  },

  configureContext(contextSize: number | null, instanceId: string, yarnEnabled?: boolean, qsaKvOffload?: QsaKvStorageSettings, kvQuantization?: KvQuantizationSettings, kvMemory?: KvMemorySettings): Promise<JobResource> {
    return request('/api/v1/runtime/context', {
      method: 'POST', body: JSON.stringify({ context_size: contextSize, instance_id: instanceId, yarn_enabled: yarnEnabled, qsa_kv_offload: qsaKvOffload, kv_quantization: kvQuantization, kv_memory: kvMemory }),
    });
  },


  clearRuntimeCache(instanceId?: string): Promise<RuntimeStatus & { released_snapshots: number }> {
    return request('/api/v1/runtime/cache/clear', {
      method: 'POST',
      body: JSON.stringify({ instance_id: instanceId }),
    });
  },


  trimRuntimeCache(
    targetBytes = 0,
    instanceId?: string,
  ): Promise<RuntimeStatus & { released_bytes: number; target_bytes: number }> {
    return request('/api/v1/runtime/cache/trim', {
      method: 'POST',
      body: JSON.stringify({ target_bytes: targetBytes, instance_id: instanceId }),
    });
  },
};
