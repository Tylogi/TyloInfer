import type { HubDownloadReuseRequest } from '../../shared/api/types';
import { modelsApi } from '../../shared/api/resources/models';
import { useDownloadReuseView } from './useDownloadReuseView';
export type { DownloadReuse } from './useDownloadReuseView';
export function useDownloadReuse(payload: HubDownloadReuseRequest | null, refreshKey: string) {
  return useDownloadReuseView(payload, refreshKey, modelsApi.downloadReuse);
}
