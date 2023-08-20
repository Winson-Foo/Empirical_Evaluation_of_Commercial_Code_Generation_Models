// To improve the maintainability of the codebase, we can make a few adjustments:

// 1. Separate the utility functions into a separate file/module to improve code organization and modularity.
// 2. Rename the utility functions to be more descriptive and aligned with their purpose.
// 3. Add proper TypeScript types and interfaces to enhance code readability and maintainability.
// 4. Move the boundary check logic into a separate function for reusability.
// 5. Remove unnecessary comments and unnecessary checks.

// Here is the refactored code:

// floodFillUtils.js

/**
 * Utility function to check if a pixel location is within the bounds of a given image.
 *
 * @param rgbData The image data.
 * @param location The pixel location.
 * @returns A boolean indicating if the location is within bounds.
 */
function isLocationWithinBounds(rgbData: number[][], location: [number, number]): boolean {
  const [x, y] = location;
  return x >= 0 && x < rgbData.length && y >= 0 && y < rgbData[0].length;
}

/**
 * Utility function to apply flood fill algorithm using a breadth-first approach using a queue.
 *
 * @param rgbData The image data to which the algorithm is applied.
 * @param location The start location on the image.
 * @param targetColor The old color to be replaced.
 * @param replacementColor The new color to replace the old one.
 */
export function breadthFirstSearch(rgbData: number[][], location: [number, number], targetColor: number, replacementColor: number): void {
  if (!isLocationWithinBounds(rgbData, location)) {
    throw new Error('Location should point to a pixel within the image');
  }

  const queue: [number, number][] = [];
  queue.push(location);

  while (queue.length > 0) {
    breadthFirstFill(rgbData, targetColor, replacementColor, queue);
  }
}

/**
 * Utility function to apply flood fill algorithm using a depth-first approach using recursion.
 *
 * @param rgbData The image data to which the algorithm is applied.
 * @param location The start location on the image.
 * @param targetColor The old color to be replaced.
 * @param replacementColor The new color to replace the old one.
 */
export function depthFirstSearch(rgbData: number[][], location: [number, number], targetColor: number, replacementColor: number): void {
  if (!isLocationWithinBounds(rgbData, location)) {
    throw new Error('Location should point to a pixel within the image');
  }

  depthFirstFill(rgbData, location, targetColor, replacementColor);
}

/**
 * Utility function to implement the breadth-first fill loop.
 *
 * @param rgbData The image data to which the algorithm is applied.
 * @param targetColor The old color to be replaced.
 * @param replacementColor The new color to replace the old one.
 * @param queue The locations that still need to be visited.
 */
function breadthFirstFill(rgbData: number[][], targetColor: number, replacementColor: number, queue: [number, number][]): void {
  const currentLocation = queue.shift();

  if (rgbData[currentLocation[0]][currentLocation[1]] === targetColor) {
    rgbData[currentLocation[0]][currentLocation[1]] = replacementColor;

    for (const [dx, dy] of neighbors) {
      const x = currentLocation[0] + dx;
      const y = currentLocation[1] + dy;
      const location = [x, y];

      if (isLocationWithinBounds(rgbData, location)) {
        queue.push(location);
      }
    }
  }
}

/**
 * Utility function to implement the depth-first fill loop.
 *
 * @param rgbData The image data to which the algorithm is applied.
 * @param location The start location on the image.
 * @param targetColor The old color to be replaced.
 * @param replacementColor The new color to replace the old one.
 */
function depthFirstFill(rgbData: number[][], location: [number, number], targetColor: number, replacementColor: number): void {
  if (rgbData[location[0]][location[1]] === targetColor) {
    rgbData[location[0]][location[1]] = replacementColor;

    for (const [dx, dy] of neighbors) {
      const x = location[0] + dx;
      const y = location[1] + dy;
      const nextLocation = [x, y];

      if (isLocationWithinBounds(rgbData, nextLocation)) {
        depthFirstFill(rgbData, nextLocation, targetColor, replacementColor);
      }
    }
  }
}

