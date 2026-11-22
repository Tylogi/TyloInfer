import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AudioDevices } from './AudioDevices';
import { RealtimeAudioController } from './RealtimeAudioController';
import { base64ToFloat32 } from './audioCodec';
import type { RealtimeCallbacks, RealtimeSessionConfig } from './realtimeTypes';

class ControlledWebSocket {
  static readonly OPEN = 1;
  static readonly CLOSED = 3;
  static instances: ControlledWebSocket[] = [];

  readonly sent: Record<string, unknown>[] = [];
  closeCalls = 0;
  readyState = ControlledWebSocket.OPEN;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;

  constructor(readonly url: string) {
    ControlledWebSocket.instances.push(this);
  }

  send(value: string): void {
    this.sent.push(JSON.parse(value) as Record<string, unknown>);
  }

  close(): void {
    if (this.readyState === ControlledWebSocket.CLOSED) return;
    this.closeCalls += 1;
    this.readyState = ControlledWebSocket.CLOSED;
    this.onclose?.();
  }

  emit(event: Record<string, unknown>): void {
    this.onmessage?.({ data: JSON.stringify(event) } as MessageEvent);
  }

  fail(): void {
    this.onerror?.();
  }
}

const config = (sessionId: string): RealtimeSessionConfig => ({
  sessionId,
  systemPrompt: 'test prompt',
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  repetitionPenalty: 1.1,
});

function callbacks(): RealtimeCallbacks {
  return {
    onState: vi.fn(),
    onLevel: vi.fn(),
    onText: vi.fn(),
    onInputStart: vi.fn(),
    onInputEnd: vi.fn(),
    onTurn: vi.fn(),
    onError: vi.fn(),
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => {
    resolve = complete;
  });
  return { promise, resolve };
}

async function finishHandshake(socket: ControlledWebSocket): Promise<void> {
  socket.emit({ type: 'session.queue_done' });
  socket.emit({ type: 'session.created' });
  await Promise.resolve();
}

describe('RealtimeAudioController', () => {
  let capturing = false;
  let inputCallback: ((samples: Float32Array) => void) | null = null;
  let startCapture: ReturnType<typeof vi.spyOn>;
  let stopCapture: ReturnType<typeof vi.spyOn>;
  let closeAudio: ReturnType<typeof vi.spyOn>;
  let controllers: RealtimeAudioController[] = [];

  function createController(events: RealtimeCallbacks, playback = true, fullDuplex = true) {
    const controller = new RealtimeAudioController(events, playback, fullDuplex);
    controllers.push(controller);
    return controller;
  }

  beforeEach(() => {
    capturing = false;
    inputCallback = null;
    controllers = [];
    ControlledWebSocket.instances = [];
    vi.stubGlobal('WebSocket', ControlledWebSocket);
    startCapture = vi
      .spyOn(AudioDevices.prototype, 'startCapture')
      .mockImplementation(async (onSamples) => {
        inputCallback = onSamples;
        capturing = true;
      });
    stopCapture = vi
      .spyOn(AudioDevices.prototype, 'stopCapture')
      .mockImplementation(async () => {
        capturing = false;
        inputCallback = null;
      });
    vi.spyOn(AudioDevices.prototype, 'capturing', 'get').mockImplementation(() => capturing);
    vi.spyOn(AudioDevices.prototype, 'ensureOutput').mockResolvedValue(undefined);
    vi.spyOn(AudioDevices.prototype, 'stopPlayback').mockImplementation(() => undefined);
    closeAudio = vi
      .spyOn(AudioDevices.prototype, 'close')
      .mockImplementation(async () => undefined);
  });

  function emitSamples(samples: Float32Array): void {
    if (!inputCallback) throw new Error('capture callback was not installed');
    inputCallback(samples);
  }

  afterEach(async () => {
    await Promise.all(controllers.map((controller) => controller.stop()));
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('stop releases the active socket and audio resources and returns to idle', async () => {
    const events = callbacks();
    const controller = createController(events);

    await controller.start(config('session-a'));
    const socket = ControlledWebSocket.instances[0];
    await finishHandshake(socket);
    await controller.stop();

    expect(socket.closeCalls).toBe(1);
    expect(stopCapture).toHaveBeenCalledOnce();
    expect(closeAudio).toHaveBeenCalledOnce();
    expect(controller.active).toBe(false);
    expect(controller.capturing).toBe(false);
    expect(events.onState).toHaveBeenLastCalledWith('idle');
  });

  it('repeated stop is idempotent and does not publish a second terminal state', async () => {
    const events = callbacks();
    const controller = createController(events);

    await controller.start(config('session-a'));
    const socket = ControlledWebSocket.instances[0];
    await finishHandshake(socket);
    await controller.stop();
    const idleStates = (events.onState as ReturnType<typeof vi.fn>).mock.calls.filter(
      ([state]) => state === 'idle',
    ).length;

    await controller.stop();

    expect(socket.closeCalls).toBe(1);
    expect(stopCapture).toHaveBeenCalledOnce();
    expect(closeAudio).toHaveBeenCalledOnce();
    expect(
      (events.onState as ReturnType<typeof vi.fn>).mock.calls.filter(
        ([state]) => state === 'idle',
      ),
    ).toHaveLength(idleStates);
  });

  it('toggleCapture stops a full-duplex microphone without starting speech processing', async () => {
    const events = callbacks();
    const controller = createController(events, true, true);

    await controller.toggleCapture(config('session-a'));
    const socket = ControlledWebSocket.instances[0];
    await finishHandshake(socket);
    expect(controller.capturing).toBe(true);

    await controller.toggleCapture(config('session-a'));

    expect(controller.capturing).toBe(false);
    expect(controller.active).toBe(false);
    expect(socket.closeCalls).toBe(1);
    expect(events.onState).not.toHaveBeenCalledWith('processing');
    expect(events.onState).toHaveBeenLastCalledWith('idle');
  });

  it('keeps late interrupted response fragments on the original turn', async () => {
    const events = callbacks();
    const controller = createController(events, false, true);

    await controller.start(config('session-a'));
    const socket = ControlledWebSocket.instances[0];
    await finishHandshake(socket);
    emitSamples(new Float32Array(16_000).fill(0.25));
    socket.emit({ kind: 'text', response_id: 'response-a', text: 'before ' });
    emitSamples(new Float32Array(11_200));
    emitSamples(new Float32Array(16_000).fill(0.25));
    socket.emit({ kind: 'text', response_id: 'response-a', text: 'interrupt tail' });

    const turns = (events.onTurn as ReturnType<typeof vi.fn>).mock.calls.map(([turn]) => turn);
    expect(turns).toHaveLength(2);
    expect(turns[0]).toMatchObject({ sessionId: 'session-a', text: 'before ' });
    expect(turns[1]).toMatchObject({
      id: turns[0].id,
      sessionId: 'session-a',
      text: 'before interrupt tail',
    });
    expect(events.onText).not.toHaveBeenLastCalledWith('session-a', 'before interrupt tail');

    await controller.stop();
  });

  it('ignores events from a replaced socket after switching sessions', async () => {
    const events = callbacks();
    const controller = createController(events, false, true);
    const sessionA = config('session-a');
    const sessionB = config('session-b');

    await controller.start(sessionA);
    const socketA = ControlledWebSocket.instances[0];
    await finishHandshake(socketA);
    await controller.start(sessionB);
    const socketB = ControlledWebSocket.instances[1];
    await finishHandshake(socketB);
    const sentBeforeStaleEvents = socketB.sent.length;

    socketA.emit({ type: 'session.queue_done' });
    socketA.emit({ type: 'session.created' });
    socketA.emit({ kind: 'text', response_id: 'old-response', text: 'stale' });

    expect(socketA.closeCalls).toBe(1);
    expect(socketB.sent).toHaveLength(sentBeforeStaleEvents);
    expect(events.onText).not.toHaveBeenCalledWith('session-b', 'stale');
    expect(events.onTurn).not.toHaveBeenCalled();

    await controller.stop();
  });

  it('surfaces device and connection failures while allowing a later restart', async () => {
    const events = callbacks();
    const controller = createController(events, false, true);
    startCapture
      .mockRejectedValueOnce(new Error('microphone denied'))
      .mockImplementation(async () => {
        capturing = true;
      });

    await controller.start(config('session-a'));
    await expect
      .poll(() => (events.onState as ReturnType<typeof vi.fn>).mock.lastCall?.[0])
      .toBe('idle');
    expect(events.onError).toHaveBeenCalledWith('microphone denied');
    expect(controller.active).toBe(false);

    await controller.start(config('session-a'));
    const socket = ControlledWebSocket.instances[0];
    socket.fail();
    expect(events.onError).toHaveBeenCalledWith('Voice connection failed');

    await controller.stop();
    await controller.submitText('retry', config('session-a'));
    expect(ControlledWebSocket.instances).toHaveLength(2);
  });

  it('coalesces repeated start calls while the first capture permission is pending', async () => {
    const events = callbacks();
    const controller = createController(events, false, true);
    const capture = deferred<void>();
    startCapture.mockImplementation(async () => {
      await capture.promise;
      capturing = true;
    });

    const first = controller.start(config('session-a'));
    await expect.poll(() => startCapture.mock.calls.length).toBe(1);
    const second = controller.start(config('session-a'));
    await Promise.resolve();
    expect(startCapture).toHaveBeenCalledOnce();

    capture.resolve();
    await Promise.all([first, second]);

    expect(ControlledWebSocket.instances).toHaveLength(1);
    expect(controller.active).toBe(true);
  });

  it('does not let a delayed capture start from an older session claim the new socket', async () => {
    const events = callbacks();
    const controller = createController(events, false, true);
    const captureA = deferred<void>();
    const captureB = deferred<void>();
    startCapture
      .mockImplementationOnce(async () => {
        await captureA.promise;
        capturing = true;
      })
      .mockImplementationOnce(async () => {
        await captureB.promise;
        capturing = true;
      });

    const startingA = controller.start(config('session-a'));
    await expect.poll(() => startCapture.mock.calls.length).toBe(1);
    const startingB = controller.start(config('session-b'));
    await expect.poll(() => startCapture.mock.calls.length).toBe(2);

    captureA.resolve();
    await Promise.resolve();
    expect(ControlledWebSocket.instances).toHaveLength(0);

    captureB.resolve();
    await Promise.all([startingA, startingB]);
    expect(ControlledWebSocket.instances).toHaveLength(1);
    expect(controller.active).toBe(true);
  });

  it('passes captured samples through the installed callback and preserves the configured session turn', async () => {
    const events = callbacks();
    const controller = createController(events, false, true);
    await controller.start(config('session-a'));
    const socket = ControlledWebSocket.instances[0];
    await finishHandshake(socket);

    expect(socket.sent[0]).toEqual({
      type: 'session.init',
      payload: {
        system_prompt: 'test prompt',
        config: {
          temperature: 0.7,
          top_p: 0.9,
          top_k: 40,
          text_repetition_penalty: 1.1,
        },
      },
    });

    const samples = new Float32Array(16_000).fill(0.25);
    emitSamples(samples);
    const input = socket.sent.find((message) => message.type === 'input.append');
    expect(input).toBeDefined();
    expect(input?.input).toMatchObject({ max_new_speak_tokens: 20 });
    expect(base64ToFloat32((input?.input as { audio: string }).audio)).toEqual(samples);
    expect(events.onInputStart).toHaveBeenCalledOnce();

    socket.emit({ kind: 'text', response_id: 'response-a', text: 'hello' });
    socket.emit({ type: 'response.step.done', response_id: 'response-a', end_of_turn: true });

    expect(events.onTurn).toHaveBeenCalledWith(
      expect.objectContaining({ sessionId: 'session-a', text: 'hello' }),
    );
    await controller.stop();
  });

  it('ends one full-duplex speech turn only after sustained silence', async () => {
    const events = callbacks();
    const controller = createController(events, false, true);
    await controller.start(config('session-a'));
    const socket = ControlledWebSocket.instances[0];
    await finishHandshake(socket);

    emitSamples(new Float32Array(8_000).fill(0.25));
    emitSamples(new Float32Array(8_000).fill(0.25));
    expect(events.onInputStart).toHaveBeenCalledOnce();
    expect(events.onInputEnd).not.toHaveBeenCalled();

    emitSamples(new Float32Array(4_000));
    expect(events.onInputEnd).not.toHaveBeenCalled();
    emitSamples(new Float32Array(8_000).fill(0.25));
    expect(events.onInputEnd).not.toHaveBeenCalled();
    emitSamples(new Float32Array(11_200));
    expect(events.onInputEnd).toHaveBeenCalledOnce();
    const startTurn = (events.onInputStart as ReturnType<typeof vi.fn>).mock.calls[0][0];
    const endTurn = (events.onInputEnd as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(endTurn.id).toBe(startTurn.id);
    expect(events.onInputEnd).toHaveBeenCalledWith(
      expect.objectContaining({ sessionId: 'session-a', audio: expect.any(Blob) }),
    );
    expect(endTurn.audio.size).toBeGreaterThan(44);

    emitSamples(new Float32Array(11_200));
    expect(events.onInputStart).toHaveBeenCalledOnce();
    expect(events.onInputEnd).toHaveBeenCalledOnce();
    await controller.stop();
  });

  it('flushes half-duplex capture with forceListen then forceSpeak and drains to idle', async () => {
    const events = callbacks();
    const controller = createController(events, false, false);
    await controller.start(config('session-a'));
    const socket = ControlledWebSocket.instances[0];
    await finishHandshake(socket);

    emitSamples(new Float32Array(16_000).fill(0.25));
    expect(socket.sent.filter((message) => message.type === 'input.append')).toHaveLength(0);
    emitSamples(new Float32Array(16_000).fill(0.5));
    const inputMessages = () => socket.sent.filter((message) => message.type === 'input.append');
    expect(inputMessages()[0]?.input).toMatchObject({ force_listen: true });

    await controller.toggleCapture(config('session-a'));
    expect(stopCapture).toHaveBeenCalledOnce();
    expect(events.onInputEnd).toHaveBeenCalledWith(
      expect.objectContaining({ sessionId: 'session-a', audio: expect.any(Blob) }),
    );
    expect(inputMessages()[1]?.input).toMatchObject({ force_speak: true });
    expect(events.onState).toHaveBeenLastCalledWith('processing');

    socket.emit({ type: 'response.step.done', response_id: 'listen', end_of_turn: false });
    socket.emit({ type: 'response.step.done', response_id: 'speak', end_of_turn: false });
    expect(inputMessages()).toHaveLength(3);
    expect(inputMessages()[2]?.input).not.toHaveProperty('force_listen');
    expect(inputMessages()[2]?.input).not.toHaveProperty('force_speak');

    socket.emit({ type: 'response.step.done', response_id: 'drain', end_of_turn: true });
    expect(events.onState).toHaveBeenLastCalledWith('idle');
    expect(controller.active).toBe(true);
    await controller.stop();
    expect(controller.active).toBe(false);
  });
});
