// Service shapes the shared display layer is handed by each application.
// Studio implements them with HTTP and its own stores; the plugin Client
// implements them with host RPC. The shared layer only sees these shapes.

import type { JobResource, ModelCacheProfile, RuntimeInstance, RuntimeStatus } from './api';
import type { KvMemorySettings, KvQuantizationSettings, QsaKvStorageSettings } from './runtime';

export interface SharedToast {
  success: (message: string, options?: { title?: string; duration?: number; id?: string }) => string;
  warning: (message: string, options?: { title?: string; duration?: number; id?: string }) => string;
  error: (message: string, options?: { title?: string; duration?: number; id?: string }) => string;
}

export type UseModelCacheProfiles = (models: string[], enabled?: boolean) => {
  cacheProfiles: Record<string, ModelCacheProfile | null>;
  cacheErrors: Record<string, string>;
};

export interface SharedRuntimeContext {
  instances: RuntimeInstance[];
  runtime: RuntimeStatus | null;
  ready: boolean;
  connectionRevision: number;
  reloadingInstances: Record<string, number>;
  reloadModelContext: (instanceId: string, contextSize: number | null, yarnEnabled?: boolean,
    qsaKvOffload?: QsaKvStorageSettings, kvQuantization?: KvQuantizationSettings, kvMemory?: KvMemorySettings) => Promise<JobResource>;
  addJob: (job: JobResource) => void;
}
