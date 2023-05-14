export type KoeiroParam = {
  speakerX: number;
  speakerY: number;
};

export const DEFAULT_PARAM: KoeiroParam = {
  speakerX: 1.32,
  speakerY: 1.88,
};

export const PRESETS: Record<string, KoeiroParam> = {
  A: {
    speakerX: -1.27,
    speakerY: 1.92,
  } as KoeiroParam,
  B: {
    speakerX: 1.32,
    speakerY: 1.88,
  } as KoeiroParam,
  C: {
    speakerX: 0.73,
    speakerY: -1.09,
  } as KoeiroParam,
  D: {
    speakerX: -0.89,
    speakerY: -2.6,
  } as KoeiroParam,
};