// To improve the maintainability of this codebase, we can make the following refactors:

// 1. Rename the function `hanoiTower` to `hanoiTowerSolver` to provide a more descriptive and accurate name for the function.

// 2. Extract the recursive logic into a separate helper function named `hanoiTowerRecursive`. This will improve readability and separate the concerns of the main function.

// 3. Remove the unnecessary comments that don't add much value to the code readability.

// 4. Format the code using an appropriate coding style for improved readability.

// Here is the refactored code:

// ```javascript
import Stack from '../../../data-structures/stack/Stack';

/**
 * Recursive helper function to solve the Tower of Hanoi puzzle.
 * @param {number} numberOfDiscs
 * @param {Stack} fromPole
 * @param {Stack} withPole
 * @param {Stack} toPole
 * @param {function(disc: number, fromPole: number[], toPole: number[])} moveCallback
 */
function hanoiTowerRecursive({
  numberOfDiscs,
  fromPole,
  withPole,
  toPole,
  moveCallback,
}) {
  if (numberOfDiscs === 1) {
    moveCallback(fromPole.peek(), fromPole.toArray(), toPole.toArray());
    const disc = fromPole.pop();
    toPole.push(disc);
  } else {
    hanoiTowerRecursive({
      numberOfDiscs: numberOfDiscs - 1,
      fromPole,
      withPole: toPole,
      toPole: withPole,
      moveCallback,
    });

    hanoiTowerRecursive({
      numberOfDiscs: 1,
      fromPole,
      withPole,
      toPole,
      moveCallback,
    });

    hanoiTowerRecursive({
      numberOfDiscs: numberOfDiscs - 1,
      fromPole: withPole,
      withPole: fromPole,
      toPole,
      moveCallback,
    });
  }
}

/**
 * Solve the Tower of Hanoi puzzle using stacks.
 * @param {number} numberOfDiscs
 * @param {function(disc: number, fromPole: number[], toPole: number[])} moveCallback
 * @param {Stack} [fromPole]
 * @param {Stack} [withPole]
 * @param {Stack} [toPole]
 */
export default function hanoiTowerSolver({
  numberOfDiscs,
  moveCallback,
  fromPole = new Stack(),
  withPole = new Stack(),
  toPole = new Stack(),
}) {
  for (let discSize = numberOfDiscs; discSize > 0; discSize -= 1) {
    fromPole.push(discSize);
  }

  hanoiTowerRecursive({
    numberOfDiscs,
    fromPole,
    withPole,
    toPole,
    moveCallback,
  });
}
// ```

// Note: In the refactored code, I made the assumption that the `Stack` class is correctly implemented and doesn't need any modifications.

