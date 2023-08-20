// To improve the maintainability of the codebase, you can make the following changes:

// 1. Add class methods instead of arrow functions for `volume` and `surfaceArea` to improve readability and consistency.

// 2. Break down the formulas into smaller, more manageable steps to improve maintainability and code understandability.

// Here's the refactored code:

// ```javascript
/**
 * This class represents a sphere and can calculate its volume and surface area
 * @constructor
 * @param {number} radius - The radius of the sphere
 * @see https://en.wikipedia.org/wiki/Sphere
 */
export default class Sphere {
  constructor(radius) {
    this.radius = radius;
  }

  volume() {
    const powerOfThree = Math.pow(this.radius, 3);
    const volume = (4 / 3) * Math.PI * powerOfThree;
    return volume;
  }

  surfaceArea() {
    const powerOfTwo = Math.pow(this.radius, 2);
    const surfaceArea = 4 * Math.PI * powerOfTwo;
    return surfaceArea;
  }
}
// ```

// By breaking down the calculations into smaller steps and using class methods, it becomes easier to understand and modify the code in the future.

