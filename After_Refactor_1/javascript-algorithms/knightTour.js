// To improve the maintainability of this codebase, you can make the following refactoring changes:

// 1. Break down the `knightTourRecursive` function into smaller, more focused functions. This will make the code more modular and easier to understand.

// 2. Use more descriptive variable and function names to improve code readability.

// 3. Remove unnecessary comments and redundant code.

// 4. Implement error handling and input validation to ensure the code works correctly with different input cases.

// Here is the refactored code:

// ```javascript
/**
 * @param {number[][]} chessboard
 * @param {number[]} position
 * @return {number[][]}
 */
function getPossibleMoves(chessboard, position) {
  const possibleMoves = [
    [position[0] - 1, position[1] - 2],
    [position[0] - 2, position[1] - 1],
    [position[0] + 1, position[1] - 2],
    [position[0] + 2, position[1] - 1],
    [position[0] - 2, position[1] + 1],
    [position[0] - 1, position[1] + 2],
    [position[0] + 1, position[1] + 2],
    [position[0] + 2, position[1] + 1],
  ];

  return possibleMoves.filter((move) => {
    const boardSize = chessboard.length;
    return move[0] >= 0 && move[1] >= 0 && move[0] < boardSize && move[1] < boardSize;
  });
}

/**
 * @param {number[][]} chessboard
 * @param {number[]} move
 * @return {boolean}
 */
function isMoveAllowed(chessboard, move) {
  return chessboard[move[0]][move[1]] !== 1;
}

/**
 * @param {number[][]} chessboard
 * @param {number[][]} moves
 * @return {boolean}
 */
function isBoardCompletelyVisited(chessboard, moves) {
  const totalPossibleMovesCount = chessboard.length ** 2;
  const existingMovesCount = moves.length;

  return totalPossibleMovesCount === existingMovesCount;
}

/**
 * @param {number[][]} chessboard
 * @param {number[][]} moves
 * @return {boolean}
 */
function findKnightTour(chessboard, moves) {
  if (isBoardCompletelyVisited(chessboard, moves)) {
    return true;
  }

  const lastMove = moves[moves.length - 1];
  const possibleMoves = getPossibleMoves(chessboard, lastMove);

  for (const move of possibleMoves) {
    if (isMoveAllowed(chessboard, move)) {
      moves.push(move);
      chessboard[move[0]][move[1]] = 1;

      if (findKnightTour(chessboard, moves)) {
        return true;
      }

      moves.pop();
      chessboard[move[0]][move[1]] = 0;
    }
  }

  return false;
}

/**
 * @param {number} chessboardSize
 * @return {number[][]}
 */
export default function knightTour(chessboardSize) {
  if (chessboardSize < 1) {
    throw new Error("Invalid chessboard size. Size must be at least 1.");
  }

  const chessboard = Array(chessboardSize).fill(null).map(() => Array(chessboardSize).fill(0));
  const moves = [];
  moves.push([0, 0]);
  chessboard[0][0] = 1;

  const solutionFound = findKnightTour(chessboard, moves);

  return solutionFound ? moves : [];
}
// ```

// By making these changes, the codebase becomes more structured, easier to read, and maintainable.

