import { saturate } from './saturate';

/**
 * Returns a linear interpolation value between 0 and 1 based on a value t and a range a to b.
 * @param {number} minValue - The minimum value of the range.
 * @param {number} maxValue - The maximum value of the range.
 * @param {number} t - The input value to be mapped to the range.
 * @returns {number} A value between 0 and 1 representing the linear interpolation between the range.
 */
export const getLinearInterpolationValue = (minValue: number, maxValue: number, t: number): number => {
  const range = maxValue - minValue;
  const normalizedT = (t - minValue) / range;
  return saturate(normalizedT);
};