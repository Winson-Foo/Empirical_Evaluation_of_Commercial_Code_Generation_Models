// To improve the maintainability of this codebase, we can make the following changes:

// 1. Add proper comments to explain the functionality of the code.
// 2. Use descriptive variable and parameter names.
// 3. Avoid unnecessary conditional statements or checks.
// 4. Use consistent formatting and indentation.
// 5. Handle edge cases and error cases explicitly.
// 6. Move the Binary Search algorithm explanation to a separate comment block.

// Here's the refactored code with the above improvements:

// ```
/**
 * Binary Search Algorithm
 *
 * @function binarySearch
 * @param {Array} sortedArray - Sorted array of integers
 * @param {Number} target - The integer to search for
 * @param {Number} low - The lowest index of the range to search in (default: 0)
 * @param {Number} high - The highest index of the range to search in (default: array length - 1)
 * @return {Number} - The index of the target element if found, else -1
 */
const binarySearch = (sortedArray, target, low = 0, high = sortedArray.length - 1) => {
  // Base case: If the range is invalid or the array is empty, return -1
  if (low > high || sortedArray.length === 0) {
    return -1
  }

  // Calculate the middle index
  const mid = low + Math.floor((high - low) / 2)

  // If the element at the middle index is equal to the target, return the index
  if (sortedArray[mid] === target) {
    return mid
  }

  // If the element at the middle index is greater than the target,
  // search the left subarray recursively
  if (sortedArray[mid] > target) {
    return binarySearch(sortedArray, target, low, mid - 1)
  }

  // Otherwise, search the right subarray recursively
  return binarySearch(sortedArray, target, mid + 1, high)
}

export { binarySearch }
// ```

// By making these changes, the codebase should be easier to understand and maintain.

