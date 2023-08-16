// To improve the maintainability of this codebase, we can do the following refactorings:

// 1. Extract the condition for checking if a new queen position conflicts with any other queens into a separate function called `doesConflictExist`, which takes in the new queen position and the list of queen positions as parameters. This function will return a boolean indicating whether a conflict exists or not.

// 2. Extract the logic for placing a queen into a separate function called `placeQueen`, which takes in the chessboard, row index, and column index as parameters. This function will update the chessboard with the new queen position and return the updated chessboard.

// 3. Rename the `nQueensRecursive` function to `solveNQueensRecursive` to provide a more descriptive name.

// Here's the refactored code:

// ```javascript
import QueenPosition from './QueenPosition';

/**
 * @param {QueenPosition[]} queensPositions
 * @param {number} rowIndex
 * @param {number} columnIndex
 * @return {boolean}
 */
function doesConflictExist(queensPositions, rowIndex, columnIndex) {
  const newQueenPosition = new QueenPosition(rowIndex, columnIndex);

  for (let queenIndex = 0; queenIndex < queensPositions.length; queenIndex += 1) {
    const currentQueenPosition = queensPositions[queenIndex];

    if (
      currentQueenPosition &&
      (newQueenPosition.columnIndex === currentQueenPosition.columnIndex ||
        newQueenPosition.rowIndex === currentQueenPosition.rowIndex ||
        newQueenPosition.leftDiagonal === currentQueenPosition.leftDiagonal ||
        newQueenPosition.rightDiagonal === currentQueenPosition.rightDiagonal)
    ) {
      return true;
    }
  }

  return false;
}

/**
 * @param {QueenPosition[]} queensPositions
 * @param {number} rowIndex
 * @param {number} columnIndex
 * @return {QueenPosition[]}
 */
function placeQueen(queensPositions, rowIndex, columnIndex) {
  const updatedPositions = [...queensPositions];
  updatedPositions[rowIndex] = new QueenPosition(rowIndex, columnIndex);
  return updatedPositions;
}

/**
 * @param {QueenPosition[][]} solutions
 * @param {QueenPosition[]} queensPositions
 * @param {number} queensCount
 * @param {number} rowIndex
 * @return {boolean}
 */
function solveNQueensRecursive(solutions, queensPositions, queensCount, rowIndex) {
  if (rowIndex === queensCount) {
    solutions.push(queensPositions);
    return true;
  }

  for (let columnIndex = 0; columnIndex < queensCount; columnIndex += 1) {
    if (!doesConflictExist(queensPositions, rowIndex, columnIndex)) {
      const updatedPositions = placeQueen(queensPositions, rowIndex, columnIndex);
      solveNQueensRecursive(solutions, updatedPositions, queensCount, rowIndex + 1);
      queensPositions[rowIndex] = null;
    }
  }

  return false;
}

/**
 * @param {number} queensCount
 * @return {QueenPosition[][]}
 */
export default function nQueens(queensCount) {
  const queensPositions = Array(queensCount).fill(null);
  const solutions = [];
  solveNQueensRecursive(solutions, queensPositions, queensCount, 0);
  return solutions;
} 

