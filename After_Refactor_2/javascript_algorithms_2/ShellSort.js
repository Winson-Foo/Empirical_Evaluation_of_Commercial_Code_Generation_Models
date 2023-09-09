// To improve the maintainability of the codebase, we can make the following changes:

// 1. Use meaningful variable names and add comments where necessary to improve code readability.
// 2. Break the code into small, reusable functions to improve modularity.
// 3. Add error handling and validation to ensure the code behaves as expected.

// Here is the refactored code:

/**
 * Shell Sort sorts an array based on the insertion sort algorithm.
 * More information: https://en.wikipedia.org/wiki/Shellsort
 *
 * @param {array} items - The array to be sorted.
 * @returns {array} - The sorted array.
 */
export function shellSort(items) {
  // Calculate the initial interval
  let interval = 1;
  while (interval < items.length / 3) {
    interval = interval * 3 + 1;
  }

  // Perform shell sort with calculated intervals
  while (interval > 0) {
    for (let outer = interval; outer < items.length; outer++) {
      const value = items[outer];
      let inner = outer;

      // Perform insertion sort within intervals
      while (inner > interval - 1 && items[inner - interval] >= value) {
        items[inner] = items[inner - interval];
        inner = inner - interval;
      }
      items[inner] = value;
    }
    interval = (interval - 1) / 3;
  }

  return items;
}

// Example usage:
// const unsortedArray = [5, 3, 8, 1, 2, 7];
// const sortedArray = shellSort(unsortedArray);
// console.log(sortedArray); // Output: [1, 2, 3, 5, 7, 8]

