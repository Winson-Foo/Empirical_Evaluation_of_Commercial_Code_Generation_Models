// To improve the maintainability of the codebase, we can break down the function into smaller, more manageable functions and add meaningful comments for better understanding. Here is the refactored code:

/**
 * Calculate the maximum level from the left for each terrace.
 *
 * @param {number[]} terraces
 * @return {number[]}
 */
function calculateLeftMaxLevels(terraces) {
  const leftMaxLevels = new Array(terraces.length).fill(0);
  leftMaxLevels[0] = terraces[0];

  for (let terraceIndex = 1; terraceIndex < terraces.length; terraceIndex += 1) {
    leftMaxLevels[terraceIndex] = Math.max(
      terraces[terraceIndex],
      leftMaxLevels[terraceIndex - 1],
    );
  }

  return leftMaxLevels;
}

/**
 * Calculate the maximum level from the right for each terrace.
 *
 * @param {number[]} terraces
 * @return {number[]}
 */
function calculateRightMaxLevels(terraces) {
  const rightMaxLevels = new Array(terraces.length).fill(0);
  rightMaxLevels[terraces.length - 1] = terraces[terraces.length - 1];

  for (let terraceIndex = terraces.length - 2; terraceIndex >= 0; terraceIndex -= 1) {
    rightMaxLevels[terraceIndex] = Math.max(
      terraces[terraceIndex],
      rightMaxLevels[terraceIndex + 1],
    );
  }

  return rightMaxLevels;
}

/**
 * Calculate the total amount of water trapped.
 *
 * @param {number[]} terraces
 * @param {number[]} leftMaxLevels
 * @param {number[]} rightMaxLevels
 * @return {number}
 */
function calculateWaterAmount(terraces, leftMaxLevels, rightMaxLevels) {
  let waterAmount = 0;

  for (let terraceIndex = 0; terraceIndex < terraces.length; terraceIndex += 1) {
    const currentTerraceBoundary = Math.min(
      leftMaxLevels[terraceIndex],
      rightMaxLevels[terraceIndex],
    );

    if (currentTerraceBoundary > terraces[terraceIndex]) {
      waterAmount += currentTerraceBoundary - terraces[terraceIndex];
    }
  }

  return waterAmount;
}

/**
 * DYNAMIC PROGRAMMING approach of solving Trapping Rain Water problem.
 *
 * @param {number[]} terraces
 * @return {number}
 */
export default function dpRainTerraces(terraces) {
  const leftMaxLevels = calculateLeftMaxLevels(terraces);
  const rightMaxLevels = calculateRightMaxLevels(terraces);
  const waterAmount = calculateWaterAmount(terraces, leftMaxLevels, rightMaxLevels);

  return waterAmount;
}

