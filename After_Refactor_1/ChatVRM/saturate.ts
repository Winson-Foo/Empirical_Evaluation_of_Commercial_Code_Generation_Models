const MIN_SATURATION: number = 0.0;
const MAX_SATURATION: number = 1.0;

/**
 * Ensures a value is within the valid range of saturation (0.0 to 1.0).
 * @param {number} x - The value to saturate.
 * @returns {number} The saturated value.
 */
export const saturate = (x: number): number =>
  Math.min(Math.max(x, MIN_SATURATION), MAX_SATURATION);

