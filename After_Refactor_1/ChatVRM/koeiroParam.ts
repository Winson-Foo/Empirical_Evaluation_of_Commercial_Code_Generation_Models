export type KoeiroParam = {
  speakerX: number;
  speakerY: number;
};

export const DEFAULT_PARAM: KoeiroParam = {
  speakerX: 1.32,
  speakerY: 1.88,
};

export const PRESETS = {
  A: {
    speakerX: -1.27,
    speakerY: 1.92,
  },
  B: {
    speakerX: 1.32,
    speakerY: 1.88,
  },
  C: {
    speakerX: 0.73,
    speakerY: -1.09,
  },
  D: {
    speakerX: -0.89,
    speakerY: -2.6,
  },
};

// Example usage:
console.log(DEFAULT_PARAM);
console.log(PRESETS.A.speakerX);
console.log(PRESETS.C.speakerY);

