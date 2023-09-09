// To improve the maintainability of this codebase, we can make the following changes:

// 1. Add comments: Add comments to explain the purpose and functionality of each section of code. This will make it easier for other developers to understand and maintain the code.

// 2. Use explicit parameter names: Instead of using generic names like "low" and "high", use more descriptive names that indicate their purpose, such as "startIndex" and "endIndex". This will make the code more self-explanatory.

// 3. Use a while loop instead of recursion: Recursion can be harder to understand and debug. By using a while loop instead, we can make the code more readable and maintainable.

// Here is the refactored code:

/**
 * @function binarySearch
 * @description Search for an integer inside a sorted integers array using the Binary Search algorithm.
 * @param {Integer[]} arr - Sorted array of integers
 * @param {Integer} searchValue - The integer to search for
 * @return {Integer} - Returns the index of the searchValue if found, else returns -1.
 * @see [BinarySearch](https://en.wikipedia.org/wiki/Binary_search_algorithm)
 */
const binarySearch = (arr, searchValue) => {
  let startIndex = 0
  let endIndex = arr.length - 1

  while (startIndex <= endIndex) {
    const midIndex = Math.floor((startIndex + endIndex) / 2)

    if (arr[midIndex] === searchValue) {
      return midIndex
    }

    if (arr[midIndex] > searchValue) {
      endIndex = midIndex - 1
    } else {
      startIndex = midIndex + 1
    }
  }

  return -1
}

export { binarySearch }

