// Returns a saturated value between 0 and 1
export const saturate = (value: number) => {
  const minValue = 0.0;
  const maxValue = 1.0;
  const clampedValue = Math.max(minValue, Math.min(value, maxValue)); // Clamp the value between min and max
  return clampedValue;
}; 