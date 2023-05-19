import { LipSyncAnalyzeResult } from "./lipSyncAnalyzeResult";

const TIME_DOMAIN_DATA_LENGTH = 2048;

/**
 * Provides functionality to synch lip movements with audio.
 */
export class LipSync {
  private readonly audio: AudioContext;
  private readonly analyser: AnalyserNode;
  private readonly timeDomainData: Float32Array;

  /**
   * Creates a new LipSync instance.
   * @param audio The AudioContext to be used for playing audio.
   */
  public constructor(audio: AudioContext) {
    this.audio = audio;
    this.analyser = audio.createAnalyser();
    this.timeDomainData = new Float32Array(TIME_DOMAIN_DATA_LENGTH);
  }

  /**
   * Analyzes the audio and returns the lip sync volume.
   */
  public update(): LipSyncAnalyzeResult {
    this.analyser.getFloatTimeDomainData(this.timeDomainData);
    const volume = this.calculateVolume();
    return { volume };
  }

  private calculateVolume(): number {
    let maxAmplitude = 0;
    for (let i = 0; i < TIME_DOMAIN_DATA_LENGTH; i++) {
      const amplitude = Math.abs(this.timeDomainData[i]);
      maxAmplitude = Math.max(maxAmplitude, amplitude);
    }
    return Math.min(1, maxAmplitude * 4);
  }

  /**
   * Plays audio from an ArrayBuffer.
   * @param buffer The ArrayBuffer containing the audio data.
   * @param onEnded An optional callback to be called when playback finishes.
   */
  public async playFromArrayBuffer(buffer: ArrayBuffer, onEnded?: () => void) {
    const audioBuffer = await this.audio.decodeAudioData(buffer);
    const bufferSource = this.audio.createBufferSource();
    bufferSource.buffer = audioBuffer;
    bufferSource.connect(this.audio.destination);
    bufferSource.connect(this.analyser);
    bufferSource.start();
    if (onEnded) {
      bufferSource.addEventListener("ended", onEnded);
    }
  }

  /**
   * Plays audio from a URL.
   * @param url The URL of the audio file.
   * @param onEnded An optional callback to be called when playback finishes.
   */
  public async playFromURL(url: string, onEnded?: () => void) {
    const res = await fetch(url);
    const buffer = await res.arrayBuffer();
    await this.playFromArrayBuffer(buffer, onEnded);
  }
}

