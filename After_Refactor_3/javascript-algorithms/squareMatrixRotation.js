// To improve the maintainability of this codebase, we can start by breaking down the code into smaller, more manageable functions. This will make the code easier to understand, test, and modify in the future. Here's the refactored code:

// ```
/**
 * @param {*[][]} originalMatrix
 * @return {*[][]}
 */
export default function squareMatrixRotation(originalMatrix) {
  const matrix = originalMatrix.slice();
  
  reflectDiagonal(matrix);
  reflectHorizontal(matrix);
  
  return matrix;
}

function reflectDiagonal(matrix) {
  for (let rowIndex = 0; rowIndex < matrix.length; rowIndex += 1) {
    for (let columnIndex = rowIndex + 1; columnIndex < matrix.length; columnIndex += 1) {
      swap(matrix, rowIndex, columnIndex, columnIndex, rowIndex);
    }
  }
}

function reflectHorizontal(matrix) {
  for (let rowIndex = 0; rowIndex < matrix.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < matrix.length / 2; columnIndex += 1) {
      swap(matrix, rowIndex, matrix.length - columnIndex - 1, rowIndex, columnIndex);
    }
  }
}

function swap(matrix, row1, col1, row2, col2) {
  [matrix[row1][col1], matrix[row2][col2]] = [matrix[row2][col2], matrix[row1][col1]];
}
// ```

// In the refactored code, we have created two separate functions, `reflectDiagonal` and `reflectHorizontal`, to perform the respective reflections on the matrix. Each function focuses on a specific part of the transformation logic, making the code easier to read and understand.

// We also created a helper function, `swap`, which takes care of swapping two elements in the matrix. This improves code readability and reduces the duplication of swapping logic in both reflection functions.

// By organizing the code into smaller, focused functions, we have improved the maintainability of the codebase. It will be easier to modify and extend the transformation logic in the future without impacting other parts of the code.

