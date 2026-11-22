import type { Page } from '@playwright/test';
import type { OfficialModelList } from '../src/shared/api/types';

const createdAt = '2026-09-23T00:00:00Z';
const model = 'Studio Test Model';
const modelCapabilities = {
  architecture_family: 'test',
  source: 'test',
  features: {
    text: true,
    image_input: false,
    video_input: false,
    audio_input: false,
    audio_output: false,
    full_duplex: false,
    mtp: false,
  },
};

const catalogSource = {
  provider: 'modelscope' as const, repo_id: 'example/studio-layout-test',
  url: 'https://www.modelscope.cn/models/example/studio-layout-test', available: true,
};
const catalogConfiguration = {
  status: 'recommended' as const, recommendation: 'two_stars' as const,
  required_memory_bytes: 64 * 2 ** 30, available_memory_bytes: 96 * 2 ** 30,
  reasons: ['More than half of the published precision tiers fit fully within the detected runtime memory budget.'],
};
export const officialCatalog: OfficialModelList = {
  system: { platform: 'macOS', machine: 'arm64', backend: 'metal',
    cpu_name: 'Apple M5 Max', cpu_cores: 18, gpu_names: ['Apple M5 Max'], gpu_cores: 40,
    physical_memory_bytes: 128 * 2 ** 30, runtime_memory_budget_bytes: 96 * 2 ** 30,
    memory_pools: [{ kind: 'uma', capacity_bytes: 128 * 2 ** 30, bandwidth_bytes_per_second: 614e9 }] },
  data: ['Studio Long-Context Mixture Model', 'Studio Compact Model'].map((name, index) => ({
    id: `catalog-${index}`, name, family: 'Layout fixture', architecture: 'studio_test',
    description: 'Deterministic catalog fixture for browser layout and memory-pressure checks.',
    description_zh: '用于浏览器布局与内存压力展示检查的固定目录数据。',
    parameter_label: '30B', active_parameter_label: '3B', modalities: ['text'],
    capabilities: ['reasoning'], precision_options: ['S2-L', 'S3-L', 'S4-L'],
    license: 'Apache-2.0', supports_ssd_streaming: true,
    sources: [catalogSource], selected_source: catalogSource, revision: 'test',
    downloads: 1000, likes: 25, updated_at: createdAt, configuration: catalogConfiguration,
    variants: ['S2-L', 'S3-L', 'S4-L'].map((label, tier) => ({
      id: label, label, format: 'tq' as const, files: [`studio-${label}.mfq`],
      byte_size: (64 + tier * 24) * 2 ** 30,
      configuration: { ...catalogConfiguration, required_memory_bytes: (64 + tier * 24) * 2 ** 30 },
    })),
  })),
};

interface MockOptions {
  language?: 'zh-CN' | 'en';
  theme?: 'light' | 'dark';
  preserveSettings?: boolean;
}


/** Install the console API fixture and record requests outside its supported routes. */
export async function mockStudioServer(page: Page, options: MockOptions = {}) {
  const state = { unexpected: [] as string[], requests: [] as string[] };
  await page.addInitScript(({ language, theme, preserveSettings }) => {
    if (preserveSettings && localStorage.getItem('tyloinfer.studio.generation.v1')) return;
    localStorage.setItem(
      'tyloinfer.studio.generation.v1',
      JSON.stringify({ language, theme }),
    );
  }, { language: options.language ?? 'en', theme: options.theme ?? 'light', preserveSettings: options.preserveSettings ?? false });
  await page.route('**/api/v1/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    const method = route.request().method();
    state.requests.push(`${method} ${path}`);
    const json = async (value: unknown) => route.fulfill({ json: value });
    if (path === '/api/v1/auth/policy' && method === 'GET') return json({ enabled: false, has_key: false });
    if (path === '/api/v1/hub/official') return json(officialCatalog);
    if (path === '/api/v1/hub/download-directory') return json({ directory: null, actual_directory: '/tmp/studio/models', default_directory: '/tmp/studio/models' });
    if (path === '/api/v1/runtime/memory-policy') return json({ model_limit_bytes: null, prefix_limit_bytes: null, prefix_directory: null, actual_prefix_directory: '/data/tyloinfer/prefix-cache' });
    if (path === '/api/v1/runtime/model-aliases') return json({ aliases: {} });
    if (path === '/api/v1/runtime/serving') return json({
      scope: new URL(route.request().url()).searchParams.get('scope') || 'session',
      started_at: Date.now() / 1000, activity: [],
      totals: { completed_requests: 0, prompt_tokens: 0, prefill_tokens: 0,
        completion_tokens: 0, cached_tokens: 0, prefill_tps: null, decode_tps: null,
        ttft_ms: null, cache_hit_rate: null },
    });
    if (path === '/api/v1/runtime/resources') return json({ sampled_at: 1, interval_seconds: 2,
      cpu_utilization_percent: null, gpus: [], disks: [], weights: [], memory_bandwidth_bytes_per_second: null,
      memory_bandwidth_limit_bytes_per_second: null, memory_bandwidth_utilization_percent: null });
    if (path === '/api/v1/runtime/cache/entries') return json({ directory: '/data/tyloinfer/prefix-cache',
      total_bytes: 0, total_blocks: 0, can_clear: true, offset: 0, limit: 100, data: [], blocks: [] });
    if (path === '/api/v1/runtime/context-policy') return json({ max_context_size: null,
      model_overrides: {}, model_yarn_enabled: {}, model_qsa_kv_offload: {}, model_kv_quantization: {}, fallback_context_size: 32768 });
    if (path.startsWith('/api/v1/runtime/yarn/')) return json({ supported: false,
      native_context: 262144, maximum_context: 262144, maximum_factor: 1, enabled: false, effective_factor: 1 });
    if (/^\/api\/v1\/models\/[^/]+\/cache-profile$/.test(path)) return json(null);
    if (path === '/api/v1/runtime/status')
      return json({
        runtime_state: 'ready',
        model,
        model_type: 'Test',
        model_capabilities: modelCapabilities,
        max_context: 8192,
        total_requests: 0,
        uptime_seconds: 60,
        active_requests: 0,
        sampling_defaults: {},
      });
    if (path === '/api/v1/runtime/capabilities')
      return json({
        model,
        model_type: 'Test',
        model_capabilities: modelCapabilities,
        vision_available: false,
        mtp_available: false,
        duplex_available: false,
      });
    if (path === '/api/v1/runtime/models') return json({ data: [{ id: model }] });
    if (path === '/api/v1/runtime/listener') return json({ host: '127.0.0.1', port: 8090, anthropic_port: 8091, configurable: true });
    if (path === '/api/v1/runtime/instances')
      return json({
        data: [
          {
            id: 'instance-1',
            model,
            state: 'ready',
            devices: ['cpu'],
            active_sessions: 1,
            queued_requests: 0,
          },
        ],
      });
    if (path === '/api/v1/runtime/realtime/capabilities')
      return json({ available: false, defaults: {} });
    if (path === '/api/v1/components/voice-output') return json({ available: false, ready: false });
    if (path === '/api/v1/models/directories') {
      const query = new URL(route.request().url()).searchParams;
      const directory = query.get('path') || ({ 'root-dir': '/', 'models-dir': '/models', 'empty-dir': '/models/empty' }[query.get('directory_id') || ''] ?? '/');
      const root = directory === '/';
      return json({
        current_id: root ? 'root-dir' : directory === '/models' ? 'models-dir' : 'empty-dir',
        current_name: root ? '/' : 'Models',
        current_path: directory,
        parent_id: root ? null : directory === '/models' ? 'root-dir' : 'models-dir',
        model_file_count: 0,
        data: root ? [{ id: 'models-dir', name: 'Models', model_file_count: 0 }] : [{ id: 'empty-dir', name: 'Empty directory', model_file_count: 0 }],
      });
    }
    if (
      [
        '/api/v1/models',
        '/api/v1/jobs',
        '/api/v1/runtime/metrics',
        '/api/v1/runtime/logs',
        '/api/v1/jobs/kinds',
        '/api/v1/presets',
        '/api/v1/runtime/profiles',
        '/api/v1/artifacts/lineage',
        '/api/v1/cluster/nodes',
      ].includes(path)
    )
      return json({ data: [] });
    state.unexpected.push(`${method} ${path}`);
    return route.fulfill({
      status: 404,
      json: { error: { code: 'NOT_MOCKED', message: path, retryable: false, details: {} } },
    });
  });
  return state;
}
