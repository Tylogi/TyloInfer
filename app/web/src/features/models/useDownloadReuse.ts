import type { HubDownloadReuseRequest } from '@shared/types/api';
import { modelsApi } from '../../shared/api/resources/models';
import { useDownloadReuseView } from '@shared/features/models/useDownloadReuseView';
export type { DownloadReuse } from '@shared/features/models/useDownloadReuseView';
export function useDownloadReuse(payload: HubDownloadReuseRequest | null, refreshKey: string) {
  return useDownloadReuseView(payload, refreshKey, modelsApi.downloadReuse);
}
