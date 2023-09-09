/**
 * Clamps the input value between 0.0 and 1.0.
 * @param {number} inputValue - The value to be saturated.
 * @returns {number} - The saturated value between 0.0 and 1.0.
 */
export const saturate = (inputValue: number): number => {
  const MIN_SATURATION = 0.0;
  const MAX_SATURATION = 1.0;

  // Clamp the input value between the minimum and maximum saturation values.
  const saturatedValue = Math.min(Math.max(inputValue, MIN_SATURATION), MAX_SATURATION);

  return saturatedValue;
};

