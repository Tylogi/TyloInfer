// Native route capabilities; the native server's registered routes are the only source.
const nativeCapabilities = {
  resources: true,
  analysis: true,
  runtimeConfiguration: false, // listener, cache and cluster policy surfaces are not implemented natively
  officialCatalog: true,
  hubSearch: true,
  hubReadme: true,
  downloadReuse: true,
  cacheReclaim: false, // prefix cache reclamation is not implemented natively
  modelDeletion: true,
};

export type ServerCapabilities = typeof nativeCapabilities;

/** Restrict unsupported native resources; the connection state no longer widens them. */
export function serverCapabilities(): ServerCapabilities {
  return nativeCapabilities;
}
