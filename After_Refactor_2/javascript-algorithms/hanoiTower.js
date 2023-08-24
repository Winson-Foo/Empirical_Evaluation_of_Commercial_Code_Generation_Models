// To improve the maintainability of the codebase, we can make the following refactoring changes:

// 1. Extract the recursive logic into a separate helper function.
// 2. Remove the unnecessary comments and add more descriptive ones.
// 3. Use more meaningful variable names.

// Here is the refactored code:

import Stack from '../../CONSTANT/javascript-algorithms/Stack';

/**
 * Recursive helper function for solving the Tower of Hanoi puzzle.
 * @param {number} numberOfDiscs - The number of discs to move.
 * @param {Stack} fromPole - The pole from which to move the discs.
 * @param {Stack} withPole - The auxiliary pole.
 * @param {Stack} toPole - The pole to which to move the discs.
 * @param {function(disc: number, fromPole: number[], toPole: number[])} moveCallback - The callback function called when a disc is moved.
 * @private
 */
function moveDiscs(numberOfDiscs, fromPole, withPole, toPole, moveCallback) {
  if (numberOfDiscs === 1) {
    // Base case: Move the top disc from the `fromPole` to the `toPole`.
    const disc = fromPole.pop();
    toPole.push(disc);
    moveCallback(disc, fromPole.toArray(), toPole.toArray());
  } else {
    // Recursive case: Move a tower of discs from the `fromPole` to the `toPole` using the `withPole` as an intermediary.
    
    // Move `numberOfDiscs - 1` discs from `fromPole` to `withPole`.
    moveDiscs(numberOfDiscs - 1, fromPole, toPole, withPole, moveCallback);
    
    // Move the remaining 1 disc from `fromPole` to `toPole`.
    moveDiscs(1, fromPole, withPole, toPole, moveCallback);
    
    // Move the tower of `numberOfDiscs - 1` discs from `withPole` to `toPole` using `fromPole` as an intermediary.
    moveDiscs(numberOfDiscs - 1, withPole, fromPole, toPole, moveCallback);
  }
}

/**
 * Solves the Tower of Hanoi puzzle.
 * @param {number} numberOfDiscs - The number of discs to move.
 * @param {function(disc: number, fromPole: number[], toPole: number[])} moveCallback - The callback function called when a disc is moved.
 * @param {Stack} [fromPole] - The initial pole from which to move the discs.
 * @param {Stack} [withPole] - The auxiliary pole.
 * @param {Stack} [toPole] - The pole to which to move the discs.
 */
export default function hanoiTower({
  numberOfDiscs,
  moveCallback,
  fromPole = new Stack(),
  withPole = new Stack(),
  toPole = new Stack(),
}) {
  // Create the discs and put them on the `fromPole`.
  for (let discSize = numberOfDiscs; discSize > 0; discSize--) {
    fromPole.push(discSize);
  }

  // Call the helper function to solve the Tower of Hanoi puzzle.
  moveDiscs(numberOfDiscs, fromPole, withPole, toPole, moveCallback);
}

