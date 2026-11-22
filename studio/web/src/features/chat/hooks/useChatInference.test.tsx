import { renderHook } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import type { RealtimeCapabilities, RuntimeStatus } from '../../../shared/api/types';
import { DEFAULT_SETTINGS } from '../../settings/configuration';
import { useChatInference } from './useChatInference';

const context = vi.hoisted(() => ({
  runtime: {} as RuntimeStatus,
  instances: [{ model: 'model', state: 'ready', mtp_supported: true, mtp_available: true }],
  selectedModel: 'model', models: [], capabilities: null, realtime: null as RealtimeCapabilities | null,
}));
const settings = { ...DEFAULT_SETTINGS };
vi.mock('../../../app/RuntimeProvider', () => ({ useRuntime: () => context }));
vi.mock('../../settings/SettingsProvider', () => ({ useSettings: () => ({ settings, updateSettings: vi.fn() }) }));

beforeEach(() => {
  Object.assign(settings, DEFAULT_SETTINGS);
  context.runtime = { mtp_service_enabled: true, sampling_defaults: { enable_mtp: true } };
  context.realtime = null;
});

it('disables chat MTP when the service forbids it without overwriting the chat preference', () => {
  context.runtime.mtp_service_enabled = false;
  const view = renderHook(() => useChatInference('text'));
  expect(view.result.current.mtpServiceEnabled).toBe(false);
  expect(view.result.current.mtpAvailable).toBe(false);
  expect(view.result.current.sampling.enable_mtp).toBe(false);
  expect(view.result.current.effectiveSettings.enableMtp).toBe(true);
  context.runtime.mtp_service_enabled = true;
  view.rerender();
  expect(view.result.current.mtpAvailable).toBe(true);
  expect(view.result.current.sampling.enable_mtp).toBe(true);
});

it('still honors a disabled per-chat preference when the service permits MTP', () => {
  settings.inheritModelDefaults = false;
  settings.enableMtp = false;
  const view = renderHook(() => useChatInference('text'));
  expect(view.result.current.mtpAvailable).toBe(true);
  expect(view.result.current.sampling.enable_mtp).toBe(false);
});

it('uses duplex runtime defaults while preserving the configured system prompt', () => {
  settings.systemPrompt = 'model-bound voice prompt';
  context.runtime = {
    mtp_service_enabled: true,
    duplex_sampling_defaults: {
      temperature: 0.3,
      top_p: 0.65,
      top_k: 12,
      text_repetition_penalty: 1.2,
    },
  };
  context.realtime = { available: true, defaults: { temperature: 0.9 } };

  const view = renderHook(() => useChatInference('full_duplex'));

  expect(view.result.current.effectiveSettings).toMatchObject({
    systemPrompt: 'model-bound voice prompt',
    temperature: 0.3,
    topP: 0.65,
    topK: 12,
    repetitionPenalty: 1.2,
    presencePenalty: 0,
    frequencyPenalty: 0,
  });
  expect(view.result.current.sampling).toMatchObject({
    temperature: 0.3,
    top_p: 0.65,
    top_k: 12,
    repetition_penalty: 1.2,
  });
});
