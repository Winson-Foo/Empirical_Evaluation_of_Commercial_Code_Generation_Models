// lipSyncAnalyzeResult.ts
export interface LipSyncAnalyzeResult {
  volume: number;
}

// lipSyncAnalyzer.ts
interface AnalyzeOptions {
  timeDomainDataLength: number;
  volumeThreshold: number;
}

export class LipSyncAnalyzer {
  private readonly analyser: AnalyserNode;
  private readonly timeDomainData: Float32Array;
  private readonly volumeThreshold: number;

  constructor(audio: AudioContext, options: AnalyzeOptions) {
    this.analyser = audio.createAnalyser();
    this.timeDomainData = new Float32Array(options.timeDomainDataLength);
    this.volumeThreshold = options.volumeThreshold;
  }

  public analyze(): LipSyncAnalyzeResult {
    this.analyser.getFloatTimeDomainData(this.timeDomainData);

    let volume = 0.0;
    for (let i = 0; i < this.timeDomainData.length; i++) {
      volume = Math.max(volume, Math.abs(this.timeDomainData[i]));
    }

    volume = 1 / (1 + Math.exp(-45 * volume + 5));
    if (volume < this.volumeThreshold) volume = 0;

    return { volume };
  }

  public connectTo(source: AudioNode) {
    source.connect(this.analyser);
  }
}

// lipSyncPlayer.ts
export class LipSyncPlayer {
  private readonly audio: AudioContext;

  constructor(audio: AudioContext) {
    this.audio = audio;
  }

  public async playFromArrayBuffer(
    buffer: ArrayBuffer,
    onEnded?: () => void,
    options: AnalyzeOptions = { timeDomainDataLength: 2048, volumeThreshold: 0.1 }
  ) {
    const audioBuffer = await this.audio.decodeAudioData(buffer);

    const bufferSource = this.audio.createBufferSource();
    bufferSource.buffer = audioBuffer;

    const analyzer = new LipSyncAnalyzer(this.audio, options);

    analyzer.connectTo(bufferSource);

    bufferSource.connect(this.audio.destination);
    bufferSource.start();
    if (onEnded) {
      bufferSource.addEventListener("ended", onEnded);
    }

    return analyzer;
  }

  public async playFromURL(
    url: string,
    onEnded?: () => void,
    options: AnalyzeOptions = { timeDomainDataLength: 2048, volumeThreshold: 0.1 }
  ) {
    const res = await fetch(url);
    const buffer = await res.arrayBuffer();
    return this.playFromArrayBuffer(buffer, onEnded, options);
  }
}