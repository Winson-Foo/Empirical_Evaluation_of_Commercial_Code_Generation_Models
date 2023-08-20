// To improve the maintainability of the codebase, we can do the following:

// 1. Add comments to explain the purpose and logic of each section of code.
// 2. Use descriptive variable names to make the code more self-explanatory.
// 3. Extract the core logic into a separate function for better modularity and reusability.
// 4. Format the code consistently according to a style guide (e.g., using consistent indentation).

// Here's the refactored code:

// ```javascript
function maximumNonAdjacentSum(nums) {
  // Find the maximum non-adjacent sum of the integers in the nums input list
  // :param nums: Array of Numbers
  // :return: The maximum non-adjacent sum

  if (nums.length === 0) {
    return 0;
  }

  let maxIncluding = nums[0];
  let maxExcluding = 0;

  for (let i = 1; i < nums.length; i++) {
    const num = nums[i];
    const temp = maxIncluding;
    maxIncluding = maxExcluding + num;
    maxExcluding = Math.max(temp, maxExcluding);
  }

  return Math.max(maxExcluding, maxIncluding);
}

// Example

// console.log(maximumNonAdjacentSum([1, 2, 3]));
// console.log(maximumNonAdjacentSum([1, 5, 3, 7, 2, 2, 6]));
// console.log(maximumNonAdjacentSum([-1, -5, -3, -7, -2, -2, -6]));
// console.log(maximumNonAdjacentSum([499, 500, -3, -7, -2, -2, -6]));

// export { maximumNonAdjacentSum };
// ```

// By following these improvements, the codebase becomes more readable, understandable, and maintainable.

