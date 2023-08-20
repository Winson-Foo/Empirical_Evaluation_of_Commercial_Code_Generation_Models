// To improve the maintainability of the codebase, we can make the following changes:

// 1. Add comments to improve code readability and understanding.
// 2. Rename variables and function names to be more descriptive.
// 3. Use const instead of let for variables that do not change within the scope.
// 4. Use a separate helper function for binary search instead of a recursive function.

// Here is the refactored code:

/**
 * Binary search algorithm to find the position of the key element in the array.
 *
 * @param {Array} array Array of numbers.
 * @param {Number} key Value to be searched
 * @param {Number} start Start index position of array
 * @param {Number} end End index position of array
 * @return {Number} Position of the key element
 */
function binarySearch(array, key, start, end) {
  while (start <= end) {
    const mid = Math.floor((start + end) / 2);

    if (array[mid] === key) {
      return mid;
    } else if (array[mid] < key) {
      start = mid + 1;
    } else {
      end = mid - 1;
    }
  }

  return start;
}

/**
 * Binary Insertion Sort
 *
 * @param {Array} array List to be sorted.
 * @return {Array} The sorted list.
 */
export function binaryInsertionSort(array) {
  const length = array.length;

  for (let i = 1; i < length; i += 1) {
    const key = array[i];
    const indexPosition = binarySearch(array, key, 0, i - 1);

    // Remove key from its current position
    array.splice(i, 1);
    // Insert key at the correct position
    array.splice(indexPosition, 0, key);
  }

  return array;
}

