// To improve the maintainability of the codebase, we can make the following changes:

// 1. Use meaningful variable names: Giving variables meaningful names will make the code more readable and easier to understand.

// 2. Break down complex calculations into smaller steps: By breaking down calculations into smaller steps, it becomes easier to understand and maintain.

// 3. Add comments: Adding comments to the code will help explain the purpose of certain lines or sections, making it easier for future developers to understand.

// The refactored code will look like this:

import pascalTriangle from '../../math/pascal-triangle/pascalTriangle';

/**
 * Calculates the number of unique paths in a grid.
 * @param {number} width - The width of the grid.
 * @param {number} height - The height of the grid.
 * @return {number} - The number of unique paths.
 */
export default function uniquePaths(width, height) {
  // Calculate the pascal line for the current grid
  const pascalLine = width + height - 2;

  // Calculate the position of the pascal line for the current grid
  const pascalLinePosition = Math.min(width, height) - 1;

  // Get the pascal triangle and return the corresponding number of unique paths
  return pascalTriangle(pascalLine)[pascalLinePosition];
}

// By following these practices, we have improved the readability and maintainability of the codebase.

