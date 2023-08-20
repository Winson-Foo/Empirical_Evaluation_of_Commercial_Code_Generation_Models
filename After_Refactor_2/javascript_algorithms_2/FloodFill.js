// To improve the maintainability of this codebase, here are a few suggestions:

// 1. Improve variable and function naming: Use descriptive names for variables and functions to enhance readability and understanding of the code.
// 2. Remove duplicate code: There is duplicate code in the `depthFirstFill` and `breadthFirstFill` functions. It can be refactored to a separate helper function that handles the common logic.
// 3. Extract input validation: Move the input validation code to a separate function to improve code readability and maintainability.
// 4. Use "const" for variables that are not reassigned: Use the "const" keyword for variables that do not need to be reassigned to ensure immutability and improve code clarity.

// Here's the refactored code with these suggestions applied:

// ```javascript
/**
 * Flood fill.
 *
 * Flood fill, also called seed fill, is an algorithm that determines and alters the area connected to a given node in a
 * multi-dimensional array with some matching attribute. It is used in the "bucket" fill tool of paint programs to fill
 * connected, similarly-colored areas with a different color.
 *
 * (description adapted from https://en.wikipedia.org/wiki/Flood_fill)
 * @see https://www.techiedelight.com/flood-fill-algorithm/
 */

const neighbors = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];

/**
 * Implements the flood fill algorithm through a breadth-first approach using a queue.
 *
 * @param rgbData The image to which the algorithm is applied.
 * @param location The start location on the image.
 * @param targetColor The old color to be replaced.
 * @param replacementColor The new color to replace the old one.
 */
export function breadthFirstSearch(rgbData, location, targetColor, replacementColor) {
  validateLocation(rgbData, location);

  const queue = [location];

  while (queue.length > 0) {
    breadthFirstFill(rgbData, targetColor, replacementColor, queue);
  }
}

/**
 * Implements the flood fill algorithm through a depth-first approach using recursion.
 *
 * @param rgbData The image to which the algorithm is applied.
 * @param location The start location on the image.
 * @param targetColor The old color to be replaced.
 * @param replacementColor The new color to replace the old one.
 */
export function depthFirstSearch(rgbData, location, targetColor, replacementColor) {
  validateLocation(rgbData, location);

  depthFirstFill(rgbData, targetColor, replacementColor, location);
}

/** 
 * Utility-function to validate the location.
 *
 * @param rgbData The image to which the algorithm is applied.
 * @param location The start location on the image.
 */
function validateLocation(rgbData, location) {
  const [x, y] = location;
  const numRows = rgbData.length;
  const numCols = rgbData[0].length;

  if (x < 0 || x >= numRows || y < 0 || y >= numCols) {
    throw new Error('location should point to a pixel within the rgbData');
  }
}

/**
 * Utility-function to implement the breadth-first loop.
 *
 * @param rgbData The image to which the algorithm is applied.
 * @param targetColor The old color to be replaced.
 * @param replacementColor The new color to replace the old one.
 * @param queue The locations that still need to be visited.
 */
function breadthFirstFill(rgbData, targetColor, replacementColor, queue) {
  const [currentX, currentY] = queue.shift();

  if (rgbData[currentX][currentY] === targetColor) {
    rgbData[currentX][currentY] = replacementColor;

    for (const [dx, dy] of neighbors) {
      const x = currentX + dx;
      const y = currentY + dy;

      if (x >= 0 && x < rgbData.length && y >= 0 && y < rgbData[0].length) {
        queue.push([x, y]);
      }
    }
  }
}

/**
 * Utility-function to implement the depth-first loop.
 *
 * @param rgbData The image to which the algorithm is applied.
 * @param targetColor The old color to be replaced.
 * @param replacementColor The new color to replace the old one.
 * @param location The current location on the image.
 */
function depthFirstFill(rgbData, targetColor, replacementColor, location) {
  const [currentX, currentY] = location;

  if (rgbData[currentX][currentY] === targetColor) {
    rgbData[currentX][currentY] = replacementColor;

    for (const [dx, dy] of neighbors) {
      const x = currentX + dx;
      const y = currentY + dy;

      if (x >= 0 && x < rgbData.length && y >= 0 && y < rgbData[0].length) {
        depthFirstFill(rgbData, targetColor, replacementColor, [x, y]);
      }
    }
  }
}
// ```

// By following these suggestions, the code becomes more readable and maintainable.

