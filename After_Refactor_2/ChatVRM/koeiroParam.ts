export type KoeiroParam = {
  speakerX: number;
  speakerY: number;
};

export const DEFAULT_PARAM: KoeiroParam = {
  speakerX: 1.32,
  speakerY: 1.88,
} as const;

export const generatePreset = (x: number, y: number): KoeiroParam => ({
  speakerX: x,
  speakerY: y
});

export const presets = {
  A: generatePreset(-1.27, 1.92),
  B: DEFAULT_PARAM,
  C: generatePreset(0.73, -1.09),
  D: generatePreset(-0.89, -2.6)
}; 