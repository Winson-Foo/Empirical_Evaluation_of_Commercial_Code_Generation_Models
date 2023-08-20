// To improve the maintainability of the code base, we can make the following changes:

// 1. Add comments to describe the purpose and functionality of each function.

// 2. Use descriptive variable names to make the code more readable.

// 3. Encapsulate the binary search logic into a separate helper function.

// 4. Use a temporary variable to store the sorted element instead of using splice, which can be more efficient.

// Here is the refactored code:

/**
 * Binary Search Algorithm
 *
 * Binary insertion sort is a sorting algorithm similar to insertion sort,
 * but instead of using linear search to find the position
 * where the element should be inserted, we use binary search.
 * Thus, we reduce the number of comparisons for inserting one element from O(N)
 * (Time complexity in Insertion Sort) to O(log N).
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
 * @param {Array} array List to be sorted.
 * @return {Array} The sorted list.
 */
export function binaryInsertionSort(array) {
  const totalLength = array.length;

  for (let i = 1; i < totalLength; i += 1) {
    const currentElement = array[i];
    const indexPosition = binarySearch(array, currentElement, 0, i - 1);

    for (let j = i - 1; j >= indexPosition; j -= 1) {
      array[j + 1] = array[j];
    }

    array[indexPosition] = currentElement;
  }

  return array;
}

