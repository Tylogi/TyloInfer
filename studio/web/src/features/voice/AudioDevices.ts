
import { StreamingLinearResampler } from './audioCodec';

const INPUT_RATE = 16_000;
const OUTPUT_RATE = 24_000;
const PLAYBACK_DELAY_SECONDS = 0.2;


export class AudioDevices {
  private stream: MediaStream | null = null;
  private inputContext: AudioContext | null = null;
  private outputContext: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private capture: AudioWorkletNode | null = null;
  private resampler: StreamingLinearResampler | null = null;
  private playAt = 0;
  private playing = new Set<AudioBufferSourceNode>();
  private captureGeneration = 0;
  private outputGeneration = 0;
  private closedOutputContexts = new WeakSet<AudioContext>();


  get capturing(): boolean {
    return this.inputContext !== null;
  }


  /** Ensures playback output exists; a later close invalidates pending setup. */
  async ensureOutput(): Promise<void> {
    const generation = ++this.outputGeneration;
    const existing = this.outputContext;
    const context = existing ?? new AudioContext({ sampleRate: OUTPUT_RATE });
    this.closedOutputContexts.delete(context);
    try {
      await context.resume();
    } catch (cause) {
      if (this.outputContext !== context && !this.closedOutputContexts.has(context)) {
        this.closedOutputContexts.add(context);
        await context.close().catch(() => undefined);
      }
      throw cause;
    }
    if (generation !== this.outputGeneration) {
      if (this.outputContext !== context && !this.closedOutputContexts.has(context)) {
        await context.close().catch(() => undefined);
      }
      return;
    }
    this.outputContext ??= context;
  }


  /** Starts microphone capture; stale permission/setup completions are cancelled and cleaned up. */
  async startCapture(onSamples: (samples: Float32Array) => void): Promise<void> {
    const generation = ++this.captureGeneration;
    let stream: MediaStream | null = null;
    let inputContext: AudioContext | null = null;
    let source: MediaStreamAudioSourceNode | null = null;
    let capture: AudioWorkletNode | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      if (generation !== this.captureGeneration) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      try {
        inputContext = new AudioContext({ sampleRate: INPUT_RATE });
      } catch {
        inputContext = new AudioContext();
      }
      const resampler = new StreamingLinearResampler(inputContext.sampleRate, INPUT_RATE);
      await Promise.all([inputContext.resume(), this.ensureOutput()]);
      if (generation !== this.captureGeneration) {
        stream.getTracks().forEach((track) => track.stop());
        await inputContext.close().catch(() => undefined);
        return;
      }
      await inputContext.audioWorklet.addModule('/pcm-capture-worklet.js');
      source = inputContext.createMediaStreamSource(stream);
      capture = new AudioWorkletNode(inputContext, 'pcm-capture');
      const silent = inputContext.createGain();
      silent.gain.value = 0;
      source.connect(capture).connect(silent).connect(inputContext.destination);
      capture.port.onmessage = (event: MessageEvent<Float32Array>) => {
        if (generation === this.captureGeneration && this.capture === capture) onSamples(event.data);
      };
      if (generation !== this.captureGeneration) {
        source.disconnect();
        capture.port.onmessage = null;
        capture.disconnect();
        stream.getTracks().forEach((track) => track.stop());
        await inputContext.close().catch(() => undefined);
        return;
      }
      this.stream = stream;
      this.inputContext = inputContext;
      this.resampler = resampler;
      this.source = source;
      this.capture = capture;
    } catch (cause) {
      stream?.getTracks().forEach((track) => track.stop());
      source?.disconnect();
      if (capture) {
        capture.port.onmessage = null;
        capture.disconnect();
      }
      await inputContext?.close().catch(() => undefined);
      if (generation === this.captureGeneration) await this.close();
      throw cause;
    }
  }


  convert(samples: Float32Array): Float32Array {
    return this.resampler?.push(samples) ?? new Float32Array(samples);
  }


  /** Cancels pending capture setup and releases only the current microphone resources. */
  async stopCapture(): Promise<void> {
    this.captureGeneration += 1;
    const stream = this.stream;
    const source = this.source;
    const capture = this.capture;
    const inputContext = this.inputContext;
    this.stream = null;
    this.inputContext = null;
    this.resampler = null;
    this.source = null;
    this.capture = null;
    stream?.getTracks().forEach((track) => track.stop());
    source?.disconnect();
    if (capture) capture.port.onmessage = null;
    capture?.disconnect();
    await inputContext?.close().catch(() => undefined);
  }


  play(samples: Float32Array, sampleRate: number): void {
    const context = this.outputContext;
    if (!samples.length || !context) return;
    const buffer = context.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(new Float32Array(samples), 0);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    this.playAt = Math.max(this.playAt, context.currentTime + PLAYBACK_DELAY_SECONDS);
    source.start(this.playAt);
    this.playAt += buffer.duration;
    this.playing.add(source);
    source.onended = () => {
      this.playing.delete(source);
      source.disconnect();
    };
  }


  stopPlayback(): void {
    for (const source of this.playing) {
      try {
        source.stop();
      } catch {

      }
      source.disconnect();
    }
    this.playing.clear();
    this.playAt = this.outputContext?.currentTime ?? 0;
  }


  /** Releases the current devices and invalidates pending input/output setup. */
  async close(): Promise<void> {
    this.stopPlayback();
    this.outputGeneration += 1;
    const output = this.outputContext;
    this.outputContext = null;
    if (output) this.closedOutputContexts.add(output);
    await this.stopCapture();
    await output?.close().catch(() => undefined);
    this.playAt = 0;
  }
}
