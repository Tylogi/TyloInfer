// Load-context defaults shared by the Studio web app and the plugin Client.
// The reading and persistence rules stay in each application.

export interface LoadContextDefaults {
  context_size: number | null;
  reserve_percent: 5 | 10 | 20;
  precision: number | null;
  jev_weights?: 'stream' | 'vram';
  vram_preallocated: boolean;
}
