/**
 * Splits the given array into smaller arrays of size "every".
 *
 * @param array - The array to split.
 * @param every - The size of each split array.
 * @returns An array of smaller arrays.
 */
export function arrayChunk<T>(array: ArrayLike<T>, every: number): T[][] {
  const chunks: T[][] = [];

  for (let i = 0; i < array.length; i += every) {
    chunks.push(array.slice(i, i + every));
  }

  return chunks;
}