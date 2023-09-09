// To improve the maintainability of the codebase, we can make the following changes:

// 1. Break down the code into smaller, more manageable functions.
// 2. Add comments to explain the purpose of each section of code.
// 3. Use descriptive variable and function names.
// 4. Move the base case check to the beginning of the function.
// 5. Use the strict equality operator (===) instead of the loose equality operator (==) for better code clarity.
// 6. Add type annotations to function parameters and return types.

// Here is the refactored code:

/**
 * @function binarySearch
 * @description Search for an integer inside a sorted array using the Binary Search Algorithm.
 * @param {number[]} arr - A sorted array of integers
 * @param {number} searchValue - The integer to search for
 * @param {number} low - The lower bound of the search range
 * @param {number} high - The upper bound of the search range
 * @returns {number} - The index of the searchValue in the array, or -1 if not found
 * @see [Binary Search](https://en.wikipedia.org/wiki/Binary_search_algorithm)
 */
const binarySearch = (arr, searchValue, low = 0, high = arr.length - 1) => {
  // Base case: empty array or search range is invalid
  if (high < low || arr.length === 0) {
    return -1
  }

  // Calculate the middle index
  const mid = low + Math.floor((high - low) / 2)

  // If the search value is found at the middle index, return the index
  if (arr[mid] === searchValue) {
    return mid
  }

  // If the search value is smaller than the middle element,
  // search in the left subarray
  if (arr[mid] > searchValue) {
    return binarySearch(arr, searchValue, low, mid - 1)
  }

  // Otherwise, search in the right subarray
  return binarySearch(arr, searchValue, mid + 1, high)
}

export { binarySearch }

