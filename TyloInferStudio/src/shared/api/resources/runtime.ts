
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
} from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';
import { adaptNativeInstance, type NativeRuntimeInstance } from '../nativeRuntime';

export interface RuntimeMemoryPolicy {
  total_limit_bytes?: number | null;
  effective_total_limit_bytes?: number | null;
  automatic_total_limit_bytes?: number | null;
  total_capacity_limit_bytes?: number | null;
  model_limit_bytes: number | null;
  prefix_limit_bytes: number | null;
  prefix_disk_limit_bytes?: number | null;
  prefix_ram_limit_bytes?: number | null;
  effective_prefix_ram_limit_bytes?: number;
  prefix_directory: string | null;
  actual_prefix_directory: string;
}

export interface RuntimeAuthenticationPolicy {
  enabled: boolean;
  has_key: boolean;
}

export interface RuntimeContextPolicy {
  fallback_context_size: number;
  max_context_size: number | null;
  model_overrides: Record<string, number>;
  model_yarn_enabled?: Record<string, boolean>;
  model_qsa_kv_offload?: Record<string, QsaKvStorageSettings>;
  model_kv_quantization?: Record<string, KvQuantizationSettings>;
  model_kv_memory?: Record<string, KvMemorySettings>;
}

export interface KvMemorySettings {
  placement: 'auto' | 'auto_ssd' | 'vram' | 'ram';
  allocation: 'preallocated' | 'dynamic';
}

export interface KvQuantizationSettings {
  enabled: boolean;
  bits: 2 | 2.5 | 3 | 3.5 | 4 | 6 | 8;
  algorithm: 'turboquant';
}

export interface QsaKvStorageSettings {
  enabled: boolean;
  budget_bytes: number;
  ram_budget_bytes?: number | null;
}

export interface YarnContextInfo {
  supported: boolean;
  native_context: number | null;
  maximum_context: number | null;
  maximum_factor: number | null;
  enabled: boolean;
  effective_factor: number;
}

export interface QsaKvOffloadInfo {
  supported: boolean;
  architecture: string;
  policy: QsaKvStorageSettings & { target_context: number };
  required_index_bytes: number | null;
  buffer_bytes: number;
  target_context: number;
  reason: string | null;
}

export interface PrefixCacheGroup {
  id: string;
  model_name: string | null;
  architecture: string | null;
  codec: string | null;
  context_size: number | null;
  bytes: number;
  blocks: number;
  prefixes: number;
  block_tokens: number;
  max_prefix_tokens: number;
  incomplete_prefixes: number;
  text_blocks: number;
  last_used_at: string;
}
export interface PrefixCacheBlock {
  id: string;
  parent: string;
  tokens: number;
  prefix_tokens: number;
  complete_chain: boolean;
  text_available: boolean;
  bytes: number;
  last_used_at: string;
}
export interface PrefixCacheInventory {
  directory: string;
  total_bytes: number;
  total_blocks: number;
  can_clear: boolean;
  data: PrefixCacheGroup[];
  blocks: PrefixCacheBlock[];
  offset: number;
  limit: number;
  matched_blocks?: number;
  search_text_unavailable?: boolean;
}
export interface PrefixCacheFilters {
  query: string;
  search_in: 'all' | 'text' | 'id';
  text_filter: 'all' | 'saved' | 'missing';
  chain_filter: 'all' | 'complete' | 'incomplete';
  sort: 'recent' | 'oldest' | 'length' | 'size';
}
export interface PrefixCacheText {
  available: boolean;
  text?: string;
  reason?: string;
  total_tokens?: number;
  offset?: number;
  next_offset?: number | null;
}

export const runtimeApi = {
  servingStatistics(scope: 'session' | 'alltime', signal?: AbortSignal): Promise<ServingStatistics> {
    return request(`/api/runtime/serving?scope=${scope}`, { signal });
  },
  authenticationPolicy(): Promise<RuntimeAuthenticationPolicy> {
    return request('/api/auth/policy');
  },
  configureAuthenticationPolicy(enabled: boolean, apiKey?: string): Promise<RuntimeAuthenticationPolicy> {
    return request('/api/auth/policy', { method: 'PUT', body: JSON.stringify({ enabled, ...(apiKey === undefined ? {} : { api_key: apiKey }) }) });
  },
  qsaKvOffloadInfo(instanceId: string, targetContext: number, signal?: AbortSignal): Promise<QsaKvOffloadInfo> {
    return request(`/api/runtime/qsa-kv/${encodeURIComponent(instanceId)}?target_context=${targetContext}`, { signal });
  },
  configureQsaKvOffload(instanceId: string, policy: Pick<QsaKvOffloadInfo['policy'], 'enabled' | 'budget_bytes'>): Promise<{ operation_id: string }> {
    return request('/api/runtime/qsa-kv', { method: 'PUT', body: JSON.stringify({ instance_id: instanceId, ...policy }) });
  },
  inferencePolicy(): Promise<{ mtp_enabled: boolean }> {
    return request('/api/runtime/inference-policy');
  },
  configureInferencePolicy(mtpEnabled: boolean): Promise<{ mtp_enabled: boolean }> {
    return request('/api/runtime/inference-policy', {
      method: 'PUT', body: JSON.stringify({ mtp_enabled: mtpEnabled }),
    });
  },
  prefixCacheEntries(namespace?: string, offset = 0, signal?: AbortSignal, filters?: PrefixCacheFilters): Promise<PrefixCacheInventory> {
    const query = new URLSearchParams({ offset: String(offset), limit: '100' });
    if (namespace) query.set('namespace', namespace);
    if (filters) Object.entries(filters).forEach(([key, value]) => query.set(key, value));
    return request(`/api/runtime/cache/entries?${query}`, { signal });
  },
  prefixCacheText(namespace: string, block: string, offset = 0): Promise<PrefixCacheText> {
    return request(`/api/runtime/cache/entries/${encodeURIComponent(namespace)}/${encodeURIComponent(block)}/text?offset=${offset}`);
  },
  purgePrefixCache(namespace?: string): Promise<{ released_bytes: number; removed_blocks: number; failed_blocks: number }> {
    return request('/api/runtime/cache/purge', { method: 'POST', body: JSON.stringify({ namespace: namespace ?? null }) });
  },
  runtimeResources(signal?: AbortSignal): Promise<RuntimeResources> {
    return request('/api/runtime/resources', { signal });
  },
  modelAliases(): Promise<{ aliases: Record<string, string> }> {
    return request('/api/runtime/model-aliases');
  },
  configureModelAliases(aliases: Record<string, string>): Promise<{ aliases: Record<string, string> }> {
    return request('/api/runtime/model-aliases', { method: 'PUT', body: JSON.stringify({ aliases }) });
  },
  memoryPolicy(): Promise<RuntimeMemoryPolicy> {
    return request('/api/runtime/memory-policy');
  },
  configureMemoryPolicy(policy: Partial<Pick<RuntimeMemoryPolicy, 'total_limit_bytes' | 'model_limit_bytes' | 'prefix_limit_bytes' | 'prefix_disk_limit_bytes' | 'prefix_ram_limit_bytes' | 'prefix_directory'>>): Promise<{ operation_id: string }> {
    return request('/api/runtime/memory-policy', { method: 'PUT', body: JSON.stringify(policy) });
  },
  runtimeListener(): Promise<RuntimeListener> {
    return request('/api/runtime/listener');
  },
  configureRuntimeListener(port: number, protocol: 'openai' | 'anthropic' = 'openai'): Promise<RuntimeListener> {
    return request('/api/runtime/listener', {
      method: 'PUT', body: JSON.stringify(protocol === 'openai' ? { port } : { port, protocol }),
    });
  },

  /** Read and adapt native instances while preserving unreported telemetry as unknown. */
  async runtimeInstances(): Promise<RuntimeInstance[]> {
    return (await request<{ data: NativeRuntimeInstance[] }>('/api/runtime/instances')).data.map(adaptNativeInstance);
  },

  configureInstance(instanceId: string, policy: { pinned: boolean; idle_ttl_seconds: number | null }): Promise<RuntimeInstance> {
    return request(`/api/runtime/instances/${encodeURIComponent(instanceId)}`, { method: 'PATCH', body: JSON.stringify(policy) });
  },



  async runtimeMetrics(limit = 200, signal?: AbortSignal, since?: string): Promise<RuntimeMetricSnapshot[]> {
    return (
      await request<{ data: RuntimeMetricSnapshot[] }>(`/api/runtime/metrics?limit=${limit}${since ? `&since=${encodeURIComponent(since)}` : ''}`, { signal })
    ).data;
  },


  async runtimeLogs(limit = 100, signal?: AbortSignal, after?: number): Promise<RuntimeLogEntry[]> {
    return (await request<{ data: RuntimeLogEntry[] }>(`/api/runtime/logs?limit=${limit}${after != null ? `&after=${after}` : ''}`, { signal })).data;
  },


  realtimeCapabilities(): Promise<RealtimeCapabilities> {
    return request('/api/runtime/realtime/capabilities');
  },


  voiceOutputComponent(): Promise<VoiceOutputComponentStatus> {
    return request('/api/components/voice-output');
  },


  installVoiceOutputComponent(): Promise<{ operation_id: string; status: 'accepted' }> {
    return request('/api/components/voice-output/install', { method: 'POST' });
  },


  activateVoiceOutputComponent(): Promise<{ active: boolean; reason?: string; error?: string }> {
    return request('/api/components/voice-output/activate', { method: 'POST' });
  },

  contextPolicy(): Promise<RuntimeContextPolicy> {
    return request('/api/runtime/context-policy');
  },

  configureContextPolicy(policy: Pick<RuntimeContextPolicy, 'max_context_size'>): Promise<RuntimeContextPolicy> {
    return request('/api/runtime/context-policy', { method: 'PUT', body: JSON.stringify(policy) });
  },

  reloadGlobalContextCap(): Promise<JobResource> {
    return request('/api/runtime/context', { method: 'POST', body: JSON.stringify({ apply_global_cap: true }) });
  },

  yarnContextInfo(instanceId: string): Promise<YarnContextInfo> {
    return request(`/api/runtime/yarn/${encodeURIComponent(instanceId)}`);
  },

  configureContext(contextSize: number | null, instanceId: string, yarnEnabled?: boolean, qsaKvOffload?: QsaKvStorageSettings, kvQuantization?: KvQuantizationSettings, kvMemory?: KvMemorySettings): Promise<JobResource> {
    return request('/api/runtime/context', {
      method: 'POST', body: JSON.stringify({ context_size: contextSize, instance_id: instanceId, yarn_enabled: yarnEnabled, qsa_kv_offload: qsaKvOffload, kv_quantization: kvQuantization, kv_memory: kvMemory }),
    });
  },


  clearRuntimeCache(instanceId?: string): Promise<RuntimeStatus & { released_snapshots: number }> {
    return request('/api/runtime/cache/clear', {
      method: 'POST',
      body: JSON.stringify({ instance_id: instanceId }),
    });
  },


  trimRuntimeCache(
    targetBytes = 0,
    instanceId?: string,
  ): Promise<RuntimeStatus & { released_bytes: number; target_bytes: number }> {
    return request('/api/runtime/cache/trim', {
      method: 'POST',
      body: JSON.stringify({ target_bytes: targetBytes, instance_id: instanceId }),
    });
  },
};
