import type { ArtifactRow } from './types.ts';

export const isCompanionId = (value: unknown) => typeof value === 'string' && /^[0-9a-f]{32}$/.test(value);
export const validCompanionIds = (ids: unknown) => Array.isArray(ids) && ids.length <= 16 && ids.every(isCompanionId) && new Set(ids).size === ids.length;
export const validCompanionPaths = (paths: unknown) => Array.isArray(paths) && paths.length > 0 && paths.length <= 16
  && paths.every((path: unknown) => typeof path === 'string' && path.length <= 4096
    && /^(?:\/(?!\/)|[A-Za-z]:[\\/])/.test(path) && !/[\x00-\x1f\x7f]/.test(path))
  && new Set(paths).size === paths.length;
export function registeredCompanions(artifacts: ArtifactRow[], modelId: string) {
  return artifacts.filter((item: ArtifactRow) => item.id !== modelId && isCompanionId(item.id) && item.format === 'gguf' && item.complete === true);
}
export function companionsAvailable(artifacts: ArtifactRow[], modelId: string, ids: unknown) {
  if (!validCompanionIds(ids)) return false;
  const eligible = new Set(registeredCompanions(artifacts, modelId).map((item: ArtifactRow) => item.id));
  return (ids as string[]).every((id: string) => eligible.has(id));
}
