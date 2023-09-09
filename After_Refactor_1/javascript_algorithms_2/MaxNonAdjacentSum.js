// To improve the maintainability of the codebase, I would suggest the following refactored code:

// ```
function maximumNonAdjacentSum(nums) {
  if (nums.length === 0) {
    return 0;
  }

  let maxIncluding = nums[0];
  let maxExcluding = 0;

  for (let i = 1; i < nums.length; i++) {
    const temp = maxIncluding;
    maxIncluding = maxExcluding + nums[i];
    maxExcluding = Math.max(temp, maxExcluding);
  }

  return Math.max(maxExcluding, maxIncluding);
}

export default maximumNonAdjacentSum;
// ```

// In this code, I made the following changes to improve maintainability:

// 1. Added a check for an empty `nums` array to return 0. This ensures that the function handles a potential edge case.

// 2. Replaced the `for...of` loop with a standard `for` loop. This change makes the code more readable and easier to understand since the loop index variable (`i`) is explicitly defined.

// 3. Changed the export statement to use the `export default` syntax. This is the recommended way to export a single value from a module.

// These changes improve the clarity and readability of the code, making it easier to maintain and understand in the future.

