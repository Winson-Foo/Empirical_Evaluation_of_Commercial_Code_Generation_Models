import { saturate } from './saturate';

/**
 * Returns a value between 0 and 1 based on linear interpolation between two values.
 * @param minValue The minimum value.
 * @param maxValue The maximum value.
 * @param currentValue The current value.
 * @returns A value between 0 and 1.
 */
export const linearInterpolate = (minValue: number, maxValue: number, currentValue: number): number => {
  const range = maxValue - minValue;
  const normalizedValue = (currentValue - minValue) / range;
  return saturate(normalizedValue);
};

