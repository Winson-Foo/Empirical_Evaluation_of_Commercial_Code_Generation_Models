// To improve the maintainability of the codebase, we can make the following changes:

// 1. Use meaningful variable names: Replace generic variable names like `move` with more descriptive names like `knightMove` or `possibleMove`.

// 2. Extract repetitive logic into separate functions: There are some blocks of code that can be extracted into separate functions to improve readability and reduce duplication. For example, the logic to check if a move is allowed can be extracted into a function called `isMoveAllowed`. Similarly, the logic to check if the board has been completely visited can be extracted into a function called `isBoardCompletelyVisited`.

// 3. Use constant values for chessboard cell states: Instead of hardcoding the cell state values (1 and 0), we can use constants to represent them. For example, we can define constants like `CELL_EMPTY` and `CELL_VISITED` to make the code more readable and easier to maintain.

// Here is the refactored code incorporating the above improvements:

// ```
const CELL_EMPTY = 0;
const CELL_VISITED = 1;

function getPossibleKnightMoves(chessboard, position) {
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

function isMoveAllowed(chessboard, move) {
  return chessboard[move[0]][move[1]] !== CELL_VISITED;
}

function isBoardCompletelyVisited(chessboard, moves) {
  const totalPossibleMovesCount = chessboard.length ** 2;
  const existingMovesCount = moves.length;

  return totalPossibleMovesCount === existingMovesCount;
}

function knightTourRecursive(chessboard, moves) {
  if (isBoardCompletelyVisited(chessboard, moves)) {
    return true;
  }

  const lastMove = moves[moves.length - 1];
  const possibleMoves = getPossibleKnightMoves(chessboard, lastMove);

  for (let moveIndex = 0; moveIndex < possibleMoves.length; moveIndex += 1) {
    const knightMove = possibleMoves[moveIndex];

    if (isMoveAllowed(chessboard, knightMove)) {
      moves.push(knightMove);
      chessboard[knightMove[0]][knightMove[1]] = CELL_VISITED;

      if (knightTourRecursive(chessboard, moves)) {
        return true;
      }

      moves.pop();
      chessboard[knightMove[0]][knightMove[1]] = CELL_EMPTY;
    }
  }

  return false;
}

export default function knightTour(chessboardSize) {
  const chessboard = Array(chessboardSize).fill(null).map(() => Array(chessboardSize).fill(CELL_EMPTY));
  const moves = [];

  const firstMove = [0, 0];
  moves.push(firstMove);
  chessboard[firstMove[0]][firstMove[0]] = CELL_VISITED;

  const solutionWasFound = knightTourRecursive(chessboard, moves);

  return solutionWasFound ? moves : [];
} 

