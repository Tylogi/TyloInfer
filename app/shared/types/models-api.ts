import type { ModelArtifact, ModelDeletionPlan, ModelCacheProfile, CheckpointAnalysis, ModelDirectoryList, HubModelSummary, HubModelSearchPage, HubModelInfo, HubDownloadReuseRequest, HubDownloadReuseResponse, OfficialModelList, ArtifactLineage } from './api';
import type { ModelLoadContextSettings, ModelLoadPlan, ModelLoadPlanQuery, DownloadDirectoryStatus } from './models';

export interface SharedModelsApi {
    browseHubModels(provider: HubModelSummary['provider'], query: string, options: {
        formats: ('tq' | 'gguf' | 'mlx')[];
        sort: 'relevance' | 'downloads' | 'likes' | 'updated';
        direction: 'asc' | 'desc';
        cursor?: string | null;
        signal?: AbortSignal;
    }): Promise<HubModelSearchPage>;
    downloadReuse(payload: HubDownloadReuseRequest, signal?: AbortSignal): Promise<HubDownloadReuseResponse>;
    downloadDirectory(): Promise<DownloadDirectoryStatus>;
    configureDownloadDirectory(directoryId: string | null): Promise<DownloadDirectoryStatus>;
    configureDownloadReclaim(evictPrefixCache: boolean): Promise<DownloadDirectoryStatus>;
    probeDownloadSources(sources: {
        provider: HubModelSummary['provider'];
        repo_id: string;
        revision?: string | null;
    }[], signal?: AbortSignal): Promise<{
        data: {
            provider: HubModelSummary['provider'];
            repo_id: string;
            bytes_per_second: number;
            route?: 'direct' | 'proxy';
        }[];
    }>;
    modelArtifacts(refresh?: boolean, includeComponents?: boolean): Promise<ModelArtifact[]>;
    modelDirectories(directoryId?: string | null, path?: string | null, signal?: AbortSignal): Promise<ModelDirectoryList>;
    modelArtifactDirectory(modelId: string): Promise<ModelDirectoryList>;
    modelDeletionPlan(modelId: string): Promise<ModelDeletionPlan>;
    deleteModel(modelId: string, token: string): Promise<{
        files: number;
        total_bytes: number;
    }>;
    checkpointAnalysis(modelId: string, signal?: AbortSignal): Promise<CheckpointAnalysis>;
    modelCacheProfile(modelId: string, signal?: AbortSignal, includeRecurrent?: boolean): Promise<ModelCacheProfile | null>;
    openModelDirectory(directoryId: string): Promise<{
        opened: boolean;
    }>;
    registerModelDirectory(directoryId: string): Promise<ModelArtifact[]>;
    loadModel(model: string, contextSize?: number | null, prefillChunkSize?: number | 'auto', policy?: {
        jev_weights?: 'stream' | 'vram';
        pin?: boolean;
        idle_ttl_seconds?: number | null;
        device_ids?: string[];
        context_management?: ModelLoadContextSettings;
        gguf_companions?: string[];
        dflash2_artifact_id?: string;
        replace_instance_id?: string;
    }): Promise<{
        operation_id: string;
        status: 'accepted';
    }>;
    modelLoadPlan(id: string, params: ModelLoadPlanQuery, signal?: AbortSignal): Promise<ModelLoadPlan>;
    unloadModel(instanceId: string, force?: boolean): Promise<{
        operation_id: string;
        status: 'accepted';
    }>;
    searchHubModels(provider: HubModelSummary['provider'], query: string, options?: {
        format?: 'all' | 'tq' | 'gguf' | 'mlx';
        sort?: 'downloads' | 'likes' | 'updated';
        limit?: number;
        signal?: AbortSignal;
    }): Promise<HubModelSummary[]>;
    hubModelReadme(model: Pick<HubModelInfo, 'provider' | 'repo_id' | 'revision'>, signal?: AbortSignal): Promise<{
        content: string | null;
    }>;
    officialHubModels(refresh?: boolean, signal?: AbortSignal): Promise<OfficialModelList>;
    resolveHubModel(reference: string, fallbackProvider: HubModelSummary['provider'], signal?: AbortSignal): Promise<HubModelInfo>;
    hubModelInfo(provider: HubModelSummary['provider'], repoId: string, revision?: string, signal?: AbortSignal): Promise<HubModelInfo>;
    removeWorkspaceArtifact(uri: string): Promise<Record<string, unknown>>;
    artifactLineage(limit?: number): Promise<ArtifactLineage[]>;

}
