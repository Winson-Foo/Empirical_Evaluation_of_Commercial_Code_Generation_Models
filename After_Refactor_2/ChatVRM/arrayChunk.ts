/**
 * Splits an array into sub-arrays of a given size.
 *
 * @param {ArrayLike<T>} array - The input array.
 * @param {number} size - The size of each sub-array.
 * @returns {T[][]} An array of sub-arrays.
 */
export const splitArrayIntoChunks = (array, size) => {
  const arrayLength = array.length;
  const chunks = [];

  let currentChunk = [];
  let remainingElements = 0;

  for (let i = 0; i < arrayLength; i++) {
    const element = array[i];

    if (remainingElements <= 0) {
      remainingElements = size;
      currentChunk = [];
      chunks.push(currentChunk);
    }

    currentChunk.push(element);
    remainingElements--;
  }

  return chunks;
};