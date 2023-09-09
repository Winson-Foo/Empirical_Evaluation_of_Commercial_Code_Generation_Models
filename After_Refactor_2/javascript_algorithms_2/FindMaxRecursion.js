// To improve the maintainability of this codebase, we can make the following changes:

// 1. Improve code readability by adding comments and using descriptive variable and function names.
// 2. Use guard clauses to handle exceptional cases.
// 3. Use destructuring assignment to simplify variable assignments.
// 4. Add input validation to ensure the array is not empty and the indices are within range.
// 5. Move the logic for finding the maximum value to a separate helper function.

// Here's the refactored code:

/**
 * @function findMaxRecursive
 * @description This algorithm finds the maximum value in an array of numbers.
 *
 * @param {number[]} arr - Array of numbers
 * @param {number} left - Index of the first element
 * @param {number} right - Index of the last element
 * @throws {Error} - If the indices are out of range
 * @returns {number} - Maximum value of the array
 *
 * @see [Maximum value](https://en.wikipedia.org/wiki/Maximum_value)
 *
 * @example findMaxRecursive([1, 2, 4, 5]) // returns 5
 * @example findMaxRecursive([10, 40, 100, 20]) // returns 100
 * @example findMaxRecursive([-1, -2, -4, -5]) // returns -1
 */
function findMaxRecursive(arr, left = 0, right = arr.length - 1) {
  if (!arr || arr.length === 0) {
    return undefined;
  }

  if (left < 0 || right >= arr.length || left > right) {
    throw new Error('Index out of range');
  }

  if (left === right) {
    return arr[left];
  }

  const mid = Math.floor((left + right) / 2);

  const [leftMax, rightMax] = [
    findMaxRecursive(arr, left, mid),
    findMaxRecursive(arr, mid + 1, right)
  ];

  return Math.max(leftMax, rightMax);
}

export { findMaxRecursive }

