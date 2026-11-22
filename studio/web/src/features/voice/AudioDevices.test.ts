
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { AudioDevices } from './AudioDevices';

const contexts: FakeContext[] = [];
const sources: Array<{
  start: ReturnType<typeof vi.fn>;
  stop: ReturnType<typeof vi.fn>;
  disconnect: ReturnType<typeof vi.fn>;
  buffer: unknown;
  onended: (() => void) | null;
  connect: ReturnType<typeof vi.fn>;
}> = [];
const stopTrack = vi.fn();
const getUserMedia = vi.fn();
const worklets: FakeWorkletNode[] = [];
let failWorklet = false;
let resumeGate: { promise: Promise<void> } | null = null;
let closeGate: { promise: Promise<void> } | null = null;
let resumeError: Error | null = null;
let requestedInputRateOverride: number | null = null;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => {
    resolve = complete;
  });
  return { promise, resolve };
}

class FakeWorkletNode {
  port = { onmessage: null as ((event: MessageEvent<Float32Array>) => void) | null };
  disconnect = vi.fn();
  constructor() {
    worklets.push(this);
  }
  connect(target: unknown) {
    return target;
  }
}

class FakeContext {
  sampleRate: number;
  currentTime = 1;
  destination = {};
  resume = vi.fn(async () => {
    await resumeGate?.promise;
    if (resumeError) throw resumeError;
  });
  close = vi.fn(async () => {
    await closeGate?.promise;
  });
  audioWorklet = {
    addModule: vi.fn(async () => {
      if (failWorklet) throw new Error('worklet unavailable');
    }),
  };
  constructor(options?: { sampleRate?: number }) {
    this.sampleRate =
      options?.sampleRate === 16_000 && requestedInputRateOverride !== null
        ? requestedInputRateOverride
        : options?.sampleRate ?? 48_000;
    contexts.push(this);
  }
  createBuffer(_channels: number, length: number, rate: number) {
    return { duration: length / rate, copyToChannel: vi.fn() };
  }
  createBufferSource() {
    const source = {
      start: vi.fn(),
      stop: vi.fn(),
      disconnect: vi.fn(),
      buffer: null as unknown,
      onended: null as (() => void) | null,
      connect: vi.fn(),
    };
    sources.push(source);
    return source;
  }
  createMediaStreamSource() {
    return { connect: (target: unknown) => target, disconnect: vi.fn() };
  }
  createGain() {
    return { gain: { value: 1 }, connect: vi.fn() };
  }
}

beforeEach(() => {
  contexts.length = 0;
  sources.length = 0;
  worklets.length = 0;
  failWorklet = false;
  resumeGate = null;
  closeGate = null;
  resumeError = null;
  requestedInputRateOverride = null;
  stopTrack.mockClear();
  getUserMedia.mockReset();
  getUserMedia.mockResolvedValue({ getTracks: () => [{ stop: stopTrack }] });
  vi.stubGlobal('AudioContext', FakeContext);
  vi.stubGlobal('AudioWorkletNode', FakeWorkletNode);
  vi.stubGlobal('navigator', {
    mediaDevices: { getUserMedia },
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it('连续音频块按样本时长串行排队，停止会释放所有源和输出上下文', async () => {
  const audio = new AudioDevices();
  await audio.ensureOutput();
  audio.play(new Float32Array(2400), 24000);
  audio.play(new Float32Array(4800), 24000);
  expect(sources[0].start).toHaveBeenCalledWith(1.2);
  expect(sources[1].start.mock.calls[0][0]).toBeCloseTo(1.3);
  await audio.close();
  expect(sources.every((source) => source.stop.mock.calls.length === 1)).toBe(true);
  expect(sources.every((source) => source.disconnect.mock.calls.length === 1)).toBe(true);
  expect(contexts[0].close).toHaveBeenCalledOnce();
});

it('录音工作线程加载失败时立即释放麦克风以及输入输出上下文', async () => {
  failWorklet = true;
  const audio = new AudioDevices();
  await expect(audio.startCapture(vi.fn())).rejects.toThrow('worklet unavailable');
  expect(audio.capturing).toBe(false);
  expect(stopTrack).toHaveBeenCalledOnce();
  expect(contexts).toHaveLength(2);
  expect(contexts.every((context) => context.close.mock.calls.length === 1)).toBe(true);
});

it('半双工结束采集只关闭输入设备，仍可播放服务返回的音频', async () => {
  const audio = new AudioDevices();
  await audio.startCapture(vi.fn());
  expect(audio.capturing).toBe(true);
  expect(contexts[0].sampleRate).toBe(16_000);
  await audio.stopCapture();
  expect(audio.capturing).toBe(false);
  expect(stopTrack).toHaveBeenCalledOnce();
  expect(contexts[1].close).not.toHaveBeenCalled();
  audio.play(new Float32Array(2400), 24000);
  expect(sources[0].start).toHaveBeenCalled();
  await audio.close();
});

it('真实采集设备把 48k PCM 分块转换为 16k，并保留线性插值结果', async () => {
  requestedInputRateOverride = 48_000;
  const audio = new AudioDevices();
  const rawBlocks: Float32Array[] = [];
  await audio.startCapture((samples) => rawBlocks.push(samples));

  const sampleAt = (index: number) => Math.sin(index / 17);
  const first = Float32Array.from({ length: 24_001 }, (_, index) => sampleAt(index));
  const second = Float32Array.from({ length: 23_999 }, (_, index) => sampleAt(index + 24_001));
  worklets[0].port.onmessage?.({ data: first } as MessageEvent<Float32Array>);
  worklets[0].port.onmessage?.({ data: second } as MessageEvent<Float32Array>);

  expect(rawBlocks).toHaveLength(2);
  const converted = [audio.convert(rawBlocks[0]), audio.convert(rawBlocks[1])];
  const output = Float32Array.from([...converted[0], ...converted[1]]);
  expect(output).toHaveLength(16_000);
  for (let index = 0; index < output.length; index += 1) {
    expect(output[index]).toBeCloseTo(sampleAt(index * 3), 5);
  }

  await audio.close();
});

it('旧权限请求晚到时释放旧设备且不覆盖较新的采集', async () => {
  const permissionA = deferred<MediaStream>();
  const permissionB = deferred<MediaStream>();
  const stopA = vi.fn();
  const stopB = vi.fn();
  const streamA = { getTracks: () => [{ stop: stopA }] } as unknown as MediaStream;
  const streamB = { getTracks: () => [{ stop: stopB }] } as unknown as MediaStream;
  getUserMedia.mockReset();
  getUserMedia.mockReturnValueOnce(permissionA.promise).mockReturnValueOnce(permissionB.promise);
  const samplesA = vi.fn();
  const samplesB = vi.fn();
  const audio = new AudioDevices();

  const startA = audio.startCapture(samplesA);
  await expect.poll(() => getUserMedia.mock.calls.length).toBe(1);
  await audio.stopCapture();
  const startB = audio.startCapture(samplesB);
  await expect.poll(() => getUserMedia.mock.calls.length).toBe(2);

  permissionB.resolve(streamB);
  await startB;
  expect(audio.capturing).toBe(true);
  const workletB = worklets[0];
  workletB.port.onmessage?.({ data: new Float32Array([1]) } as MessageEvent<Float32Array>);
  expect(samplesB).toHaveBeenCalledOnce();

  permissionA.resolve(streamA);
  await startA;
  expect(stopA).toHaveBeenCalledOnce();
  expect(audio.capturing).toBe(true);
  expect(contexts).toHaveLength(2);
  expect(contexts[0].close).not.toHaveBeenCalled();
  workletB.port.onmessage?.({ data: new Float32Array([2]) } as MessageEvent<Float32Array>);
  expect(samplesB).toHaveBeenCalledTimes(2);
  expect(samplesA).not.toHaveBeenCalled();
  await audio.close();
});

it('关闭时不会让延迟的输出初始化复活已关闭的上下文', async () => {
  const resume = deferred<void>();
  resumeGate = resume;
  const audio = new AudioDevices();
  const pending = audio.ensureOutput();
  await expect.poll(() => contexts.length).toBe(1);
  await audio.close();
  resume.resolve();
  await pending;
  expect(contexts[0].close).toHaveBeenCalledOnce();

  resumeGate = null;
  await audio.ensureOutput();
  expect(contexts).toHaveLength(2);
  await audio.close();
});

it('输入上下文关闭挂起时不会清掉后来接管的采集资源', async () => {
  const closing = deferred<void>();
  closeGate = closing;
  const stopA = vi.fn();
  const stopB = vi.fn();
  const streamA = { getTracks: () => [{ stop: stopA }] } as unknown as MediaStream;
  const streamB = { getTracks: () => [{ stop: stopB }] } as unknown as MediaStream;
  getUserMedia.mockReset();
  getUserMedia.mockReturnValueOnce(streamA).mockReturnValueOnce(streamB);
  const samplesB = vi.fn();
  const audio = new AudioDevices();
  await audio.startCapture(vi.fn());
  const stopping = audio.stopCapture();
  await expect.poll(() => contexts[0].close.mock.calls.length).toBe(1);
  const startingB = audio.startCapture(samplesB);
  await startingB;
  expect(audio.capturing).toBe(true);

  closing.resolve();
  await stopping;
  expect(audio.capturing).toBe(true);
  worklets[1].port.onmessage?.({ data: new Float32Array([1]) } as MessageEvent<Float32Array>);
  expect(samplesB).toHaveBeenCalledOnce();
  await audio.close();
});

it('输出上下文 resume 失败时会关闭未安装的本地上下文', async () => {
  resumeError = new Error('output resume failed');
  const audio = new AudioDevices();
  await expect(audio.ensureOutput()).rejects.toThrow('output resume failed');
  expect(contexts[0].close).toHaveBeenCalledOnce();

  resumeError = null;
  await audio.ensureOutput();
  expect(contexts).toHaveLength(2);
  await audio.close();
});
