// Returns a value between 0 and 1 representing the position of t within the range defined by a and b.
// a: lower bound of the range
// b: upper bound of the range 
// t: value to place within the range
export default function calculateLinearStepInRange(lowerBound: number, upperBound: number, value: number): number {
  const saturation = (value - lowerBound) / (upperBound - lowerBound);
  return saturate(saturation);
}