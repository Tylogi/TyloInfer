import type { RuntimeCapabilities, RuntimeStatus, RuntimeResources, RuntimeListener, RuntimeModel, RuntimeInstance, RuntimeMetricSnapshot, RuntimeLogEntry, RealtimeCapabilities, VoiceOutputComponentStatus, JobResource, ServingStatistics } from './api';
import type { RuntimeMemoryPolicy, RuntimeAuthenticationPolicy, RuntimeContextPolicy, KvMemorySettings, KvQuantizationSettings, QsaKvStorageSettings, YarnContextInfo, QsaKvOffloadInfo, PrefixCacheGroup, PrefixCacheBlock, PrefixCacheInventory, PrefixCacheFilters, PrefixCacheText } from './runtime';

export interface SharedRuntimeApi {
    servingStatistics(scope: 'session' | 'alltime', signal?: AbortSignal): Promise<ServingStatistics>;
    authenticationPolicy(): Promise<RuntimeAuthenticationPolicy>;
    configureAuthenticationPolicy(enabled: boolean, apiKey?: string): Promise<RuntimeAuthenticationPolicy>;
    qsaKvOffloadInfo(instanceId: string, targetContext: number, signal?: AbortSignal): Promise<QsaKvOffloadInfo>;
    configureQsaKvOffload(instanceId: string, policy: Pick<QsaKvOffloadInfo['policy'], 'enabled' | 'budget_bytes'>): Promise<{
        operation_id: string;
    }>;
    inferencePolicy(): Promise<{
        mtp_enabled: boolean;
    }>;
    configureInferencePolicy(mtpEnabled: boolean): Promise<{
        mtp_enabled: boolean;
    }>;
    prefixCacheEntries(namespace?: string, offset?: number, signal?: AbortSignal, filters?: PrefixCacheFilters): Promise<PrefixCacheInventory>;
    prefixCacheText(namespace: string, block: string, offset?: number): Promise<PrefixCacheText>;
    purgePrefixCache(namespace?: string): Promise<{
        released_bytes: number;
        removed_blocks: number;
        failed_blocks: number;
    }>;
    runtimeResources(signal?: AbortSignal): Promise<RuntimeResources>;
    modelAliases(): Promise<{
        aliases: Record<string, string>;
    }>;
    configureModelAliases(aliases: Record<string, string>): Promise<{
        aliases: Record<string, string>;
    }>;
    memoryPolicy(): Promise<RuntimeMemoryPolicy>;
    configureMemoryPolicy(policy: Partial<Pick<RuntimeMemoryPolicy, 'total_limit_bytes' | 'model_limit_bytes' | 'prefix_limit_bytes' | 'prefix_disk_limit_bytes' | 'prefix_ram_limit_bytes' | 'prefix_directory'>>): Promise<{
        operation_id: string;
    }>;
    runtimeListener(): Promise<RuntimeListener>;
    configureRuntimeListener(port: number, protocol?: 'openai' | 'anthropic'): Promise<RuntimeListener>;
    runtimeCapabilities(instanceId?: string | null): Promise<RuntimeCapabilities>;
    runtimeStatus(instanceId?: string | null): Promise<RuntimeStatus>;
    runtimeModels(): Promise<RuntimeModel[]>;
    runtimeInstances(): Promise<RuntimeInstance[]>;
    configureInstance(instanceId: string, policy: {
        pinned?: boolean;
        idle_ttl_seconds?: number | null;
        dflash2_enabled?: boolean;
    }): Promise<RuntimeInstance>;
    runtimeMetrics(limit?: number, signal?: AbortSignal, since?: string): Promise<RuntimeMetricSnapshot[]>;
    runtimeLogs(limit?: number, signal?: AbortSignal, after?: number): Promise<RuntimeLogEntry[]>;
    realtimeCapabilities(): Promise<RealtimeCapabilities>;
    voiceOutputComponent(): Promise<VoiceOutputComponentStatus>;
    installVoiceOutputComponent(): Promise<{
        operation_id: string;
        status: 'accepted';
    }>;
    activateVoiceOutputComponent(): Promise<{
        active: boolean;
        reason?: string;
        error?: string;
    }>;
    reloadRuntime(contextSize: number, instanceId?: string): Promise<RuntimeStatus>;
    contextPolicy(): Promise<RuntimeContextPolicy>;
    configureContextPolicy(policy: Pick<RuntimeContextPolicy, 'max_context_size'>): Promise<RuntimeContextPolicy>;
    reloadGlobalContextCap(): Promise<JobResource>;
    yarnContextInfo(instanceId: string): Promise<YarnContextInfo>;
    configureContext(contextSize: number | null, instanceId: string, yarnEnabled?: boolean, qsaKvOffload?: QsaKvStorageSettings, kvQuantization?: KvQuantizationSettings, kvMemory?: KvMemorySettings): Promise<JobResource>;
    clearRuntimeCache(instanceId?: string): Promise<RuntimeStatus & {
        released_snapshots: number;
    }>;
    trimRuntimeCache(targetBytes?: number, instanceId?: string): Promise<RuntimeStatus & {
        released_bytes: number;
        target_bytes: number;
    }>;

}
