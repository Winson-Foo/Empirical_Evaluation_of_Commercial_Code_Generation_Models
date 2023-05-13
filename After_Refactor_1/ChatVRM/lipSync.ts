import { LipSyncAnalyzeResult } from "./lipSyncAnalyzeResult";

const TIME_DOMAIN_DATA_LENGTH = 2048;
const VOLUME_THRESHOLD = 0.1;
const COOK_FACTOR_A = -45;
const COOK_FACTOR_B = 5;
const ENDED_EVENT = "ended";

export class LipSync {
  public readonly audioContext: AudioContext;
  public readonly audioAnalyser: AnalyserNode;
  public readonly audioData: Float32Array;

  public constructor(context: AudioContext) {
    this.audioContext = context;
    this.audioAnalyser = context.createAnalyser();
    this.audioData = new Float32Array(TIME_DOMAIN_DATA_LENGTH);
  }

  public update(): LipSyncAnalyzeResult {
    this.audioAnalyser.getFloatTimeDomainData(this.audioData);

    let maxVolume = 0.0;
    for (let i = 0; i < TIME_DOMAIN_DATA_LENGTH; i++) {
      maxVolume = Math.max(maxVolume, Math.abs(this.audioData[i]));
    }

    // Eq. function for "cooking" the volume
    const volume = 1 / (1 + Math.exp(COOK_FACTOR_A * maxVolume + COOK_FACTOR_B));
    return { volume: volume < VOLUME_THRESHOLD ? 0 : volume };
  }

  public async playFromBuffer(buffer: ArrayBuffer, onEnded?: () => void) {
    const audioBuffer = await this.audioContext.decodeAudioData(buffer);
    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioAnalyser);
    source.connect(this.audioContext.destination);
    if (onEnded) source.addEventListener(ENDED_EVENT, onEnded);
    source.start();
  }

  public async playFromUrl(url: string, onEnded?: () => void) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    await this.playFromBuffer(buffer, onEnded);
  }
}

