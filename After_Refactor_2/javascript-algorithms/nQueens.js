// To improve the maintainability of this codebase, you can follow the following steps:

// 1. Separate the code into smaller functions with single responsibilities to improve readability and maintainability.
// 2. Use more descriptive variable and function names to make the code easier to understand.
// 3. Remove unnecessary comments or refactor them to be more meaningful.
// 4. Use proper formatting and indentation to enhance code readability.

// Here is the refactored code:

import QueenPosition from '../../CONSTANT/javascript_algorithms/QueenPosition';

function isSafe(queensPositions, rowIndex, columnIndex) {
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
      return false;
    }
  }

  return true;
}

function cloneQueensPositions(previousQueensPositions) {
  return previousQueensPositions.map((queenPosition) => {
    return queenPosition ? new QueenPosition(queenPosition.rowIndex, queenPosition.columnIndex) : queenPosition;
  });
}

function nQueensRecursive(solutions, previousQueensPositions, queensCount, rowIndex) {
  const queensPositions = cloneQueensPositions(previousQueensPositions);

  if (rowIndex === queensCount) {
    solutions.push(queensPositions);
    return true;
  }

  for (let columnIndex = 0; columnIndex < queensCount; columnIndex += 1) {
    if (isSafe(queensPositions, rowIndex, columnIndex)) {
      queensPositions[rowIndex] = new QueenPosition(rowIndex, columnIndex);
      nQueensRecursive(solutions, queensPositions, queensCount, rowIndex + 1);
      queensPositions[rowIndex] = null;
    }
  }

  return false;
}

export default function nQueens(queensCount) {
  const queensPositions = Array(queensCount).fill(null);
  const solutions = [];
  nQueensRecursive(solutions, queensPositions, queensCount, 0);
  return solutions;
}

