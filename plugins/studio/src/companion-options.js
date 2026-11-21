export const isCompanionId = value => typeof value === 'string' && /^[0-9a-f]{32}$/.test(value);
export const validCompanionIds = ids => Array.isArray(ids) && ids.length <= 16 && ids.every(isCompanionId) && new Set(ids).size === ids.length;
export const validCompanionPaths = paths => Array.isArray(paths) && paths.length > 0 && paths.length <= 16
  && paths.every(path => typeof path === 'string' && path.length <= 4096
    && /^(?:\/(?!\/)|[A-Za-z]:[\\/])/.test(path) && !/[\x00-\x1f\x7f]/.test(path))
  && new Set(paths).size === paths.length;
export function registeredCompanions(artifacts, modelId) {
  return artifacts.filter(item => item.id !== modelId && isCompanionId(item.id) && item.format === 'gguf' && item.complete === true);
}
export function companionsAvailable(artifacts, modelId, ids) {
  if (!validCompanionIds(ids)) return false;
  const eligible = new Set(registeredCompanions(artifacts, modelId).map(item => item.id));
  return ids.every(id => eligible.has(id));
}
