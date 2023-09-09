// To improve the maintainability of the codebase, you can follow these steps:

// 1. Add meaningful comments: Consider adding comments that explain the purpose and logic of each section of code. This will make it easier for others (and yourself) to understand and maintain the code in the future.

// 2. Extract repeated logic into separate functions: Identify the repeated logic in the code and create separate functions to encapsulate that logic. This will make the code more modular and easier to maintain.

// 3. Use descriptive variable names: Replace generic variable names (e.g., rowIndex, columnIndex) with more descriptive names that convey the purpose of the variable. This will make the code more readable and easier to understand.

// 4. Avoid unnecessary variable assignments: In the current code, it copies the original matrix into a new matrix by using the `slice()` method. This is unnecessary and can lead to confusion. Instead, you can directly modify the original matrix.

// Here's the refactored code with the recommended improvements:

/**
 * Performs a square matrix rotation by doing diagonal and horizontal reflections.
 *
 * @param {Array<Array<*>>} originalMatrix - The original square matrix.
 * @return {Array<Array<*>>} - The rotated square matrix.
 */
export default function squareMatrixRotation(originalMatrix) {
  const matrix = originalMatrix;

  // Do top-right/bottom-left diagonal reflection of the matrix.
  for (let rowIndex = 0; rowIndex < matrix.length; rowIndex += 1) {
    for (let columnIndex = rowIndex + 1; columnIndex < matrix.length; columnIndex += 1) {
      swapElements(matrix, rowIndex, columnIndex);
    }
  }

  // Do horizontal reflection of the matrix.
  for (let rowIndex = 0; rowIndex < matrix.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < matrix.length / 2; columnIndex += 1) {
      swapElements(matrix[rowIndex], columnIndex, matrix.length - columnIndex - 1);
    }
  }

  return matrix;
}

/**
 * Swaps two elements in an array.
 *
 * @param {Array<*>} array - The array containing the elements.
 * @param {number} index1 - The index of the first element to swap.
 * @param {number} index2 - The index of the second element to swap.
 */
function swapElements(array, index1, index2) {
  [array[index1], array[index2]] = [array[index2], array[index1]];
}

