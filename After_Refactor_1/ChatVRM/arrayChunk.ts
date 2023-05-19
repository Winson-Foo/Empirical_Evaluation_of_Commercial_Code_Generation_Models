/**
 * ```js
 * arrayChunk( [ 1, 2, 3, 4, 5, 6 ], 2 )
 * // will be
 * [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ]
 * ```
 */
export function arrayChunk<T>(array: ArrayLike<T>, every: number): T[][] {
  const chunks: T[][] = [];
  let currentChunk: T[] = [];

  array.forEach((el, index) => {
    currentChunk.push(el);

    if (currentChunk.length === every || index === array.length - 1) {
      chunks.push(currentChunk);
      currentChunk = [];
    }
  });

  return chunks;
}