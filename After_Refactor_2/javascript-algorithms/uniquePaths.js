// To improve the maintainability of the codebase, you can apply the following changes:

// 1. Remove the import statement for `pascalTriangle` as it is not necessary.
// 2. Rename the function `uniquePaths` to a more descriptive name like `calculateUniquePaths`.
// 3. Add parameter validations to ensure that `width` and `height` are valid positive numbers.
// 4. Add comments to explain the purpose of the variables and clarify any complex logic.

// Here's the refactored code:

// ```javascript
/**
 * Calculates the number of unique paths from top-left to bottom-right
 * in a grid of given width and height.
 * @param {number} width - The width of the grid.
 * @param {number} height - The height of the grid.
 * @return {number} - The number of unique paths.
 */
export default function calculateUniquePaths(width, height) {
  // Validate parameters
  if (typeof width !== 'number' || typeof height !== 'number' || width <= 0 || height <= 0) {
    throw new Error('Width and height must be valid positive numbers.');
  }

  // Calculate the position of the line in Pascal's Triangle
  const pascalLine = width + height - 2;
  const pascalLinePosition = Math.min(width, height) - 1;

  // Retrieve the value from Pascal's Triangle
  return pascalTriangle(pascalLine)[pascalLinePosition];
}
// ```

// Note: The `pascalTriangle` function is not provided in the given code snippet, so you need to make sure it is imported or defined somewhere else in your code.

