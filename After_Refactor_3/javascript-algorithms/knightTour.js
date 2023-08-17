// To improve the maintainability of the codebase, you can follow these refactoring suggestions:

// 1. Use descriptive variable and function names to improve code readability.
// 2. Break down complex functions into smaller, more modular functions for better organization and reusability.
// 3. Use consistent code formatting and indentation to improve code readability.
// 4. Remove unnecessary comments and simplify code where possible.
// 5. Use immutable data structures and avoid mutating variables directly.
// 6. Use TypeScript or JSDoc annotations to provide better type safety and documentation.

// Below is the refactored code incorporating these suggestions:

// ```javascript
/**
 * Returns an array of all possible knight moves from the given position on the chessboard.
 * @param {number[][]} chessboard - The current state of the chessboard.
 * @param {number[]} position - The current position of the knight.
 * @returns {number[][]} - Array of possible moves.
 */
function getPossibleMoves(chessboard, position) {
  const [x, y] = position;
  const possibleMoves = [
    [x - 1, y - 2],
    [x - 2, y - 1],
    [x - 2, y + 1],
    [x - 1, y + 2],
    [x + 1, y + 2],
    [x + 2, y + 1],
    [x + 2, y - 1],
    [x + 1, y - 2],
  ];

  const boardSize = chessboard.length;
  return possibleMoves.filter(([x, y]) => x >= 0 && y >= 0 && x < boardSize && y < boardSize);
}

/**
 * Checks if a move is allowed on the chessboard.
 * @param {number[][]} chessboard - The current state of the chessboard.
 * @param {number[]} move - The move to be checked.
 * @returns {boolean} - True if the move is allowed, false otherwise.
 */
function isMoveAllowed(chessboard, move) {
  const [x, y] = move;
  return chessboard[x][y] !== 1;
}

/**
 * Checks if the chessboard has been completely visited.
 * @param {number[][]} chessboard - The current state of the chessboard.
 * @param {number[][]} moves - The list of moves made so far.
 * @returns {boolean} - True if the chessboard has been completely visited, false otherwise.
 */
function isBoardCompletelyVisited(chessboard, moves) {
  const totalPossibleMovesCount = chessboard.length ** 2;
  const existingMovesCount = moves.length;
  return totalPossibleMovesCount === existingMovesCount;
}

/**
 * Recursive function to find a knight tour on the chessboard.
 * @param {number[][]} chessboard - The current state of the chessboard.
 * @param {number[][]} moves - The list of moves made so far.
 * @returns {boolean} - True if a solution was found, false otherwise.
 */
function knightTourRecursive(chessboard, moves) {
  if (isBoardCompletelyVisited(chessboard, moves)) {
    return true;
  }

  const lastMove = moves[moves.length - 1];
  const possibleMoves = getPossibleMoves(chessboard, lastMove);

  for (const move of possibleMoves) {
    if (isMoveAllowed(chessboard, move)) {
      moves.push(move);
      chessboard[move[0]][move[1]] = 1;

      if (knightTourRecursive(chessboard, moves)) {
        return true;
      }

      moves.pop();
      chessboard[move[0]][move[1]] = 0;
    }
  }

  return false;
}

/**
 * Finds a knight tour on the chessboard of the given size.
 * @param {number} chessboardSize - The size of the chessboard.
 * @returns {number[][]} - Array of moves representing the knight tour, or an empty array if no solution was found.
 */
export default function knightTour(chessboardSize) {
  const chessboard = Array(chessboardSize).fill(null).map(() => Array(chessboardSize).fill(0));
  const moves = [];

  moves.push([0, 0]);
  chessboard[0][0] = 1;

  const solutionWasFound = knightTourRecursive(chessboard, moves);
  return solutionWasFound ? moves : [];
}
// ```

// Note: The refactored code improves maintainability by using more descriptive names, breaking down functions, simplifying logic, and applying consistent code formatting. However, without further context or requirements, it is difficult to assess if there are additional improvements that can be made.

