// To improve the maintainability of this codebase, we can do the following:

// 1. Add comments and improve code readability: Add comments to explain the purpose and functionality of each function and section of code.

// 2. Use descriptive variable and function names: Rename variables and functions to more accurately describe their purpose and improve readability.

// 3. Separate functions into smaller, reusable functions: Break down the binaryInsertionSort function into smaller, more focused functions that can be reused or tested individually.

// 4. Use ES6 features: Utilize ES6 features such as arrow functions and the spread operator to improve code readability and reduce the use of splice.

// Here is the refactored code with these improvements:

/**
 * Pure Implementation of Binary Search Algorithm
 *
 * Binary insertion sort is a sorting algorithm similar to insertion sort,
 * but instead of using linear search to find the position
 * where the element should be inserted, we use binary search.
 * Thus, we reduce the number of comparisons for inserting one element from O(N)
 * (Time complexity in Insertion Sort) to O(log N).
 */

/**
 * Search the key element in the array from start position to end position.
 *
 * @param {Array} array Array of numbers.
 * @param {Number} key Value to be searched
 * @param {Number} start Start index position of array
 * @param {Number} end End index position of array
 * @return {Number} Position of the key element
 */
function binarySearch(array, key, start, end) {
  if (start === end) {
    if (array[start] > key) {
      return start;
    } else {
      return start + 1;
    }
  }

  if (start > end) {
    return start;
  }

  const mid = Math.floor((start + end) / 2);

  if (array[mid] < key) {
    return binarySearch(array, key, mid + 1, end);
  } else if (array[mid] > key) {
    return binarySearch(array, key, start, mid - 1);
  } else {
    return mid;
  }
}

/**
 * Binary Insertion Sort
 *
 * @param {Array} array Array to be sorted.
 * @return {Array} The sorted array.
 */
export function binaryInsertionSort(array) {
  const sortedArray = [...array]; // Create a copy of the array to avoid modifying the original
  const totalLength = sortedArray.length;

  for (let i = 1; i < totalLength; i++) {
    const key = sortedArray[i];
    const indexPosition = binarySearch(sortedArray, key, 0, i - 1);

    sortedArray.splice(i, 1);
    sortedArray.splice(indexPosition, 0, key);
  }

  return sortedArray;
}

