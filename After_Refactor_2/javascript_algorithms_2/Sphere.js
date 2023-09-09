// To improve the maintainability of the codebase, you can make the following changes:

// 1. Remove the arrow functions and use regular function syntax for the methods.
// 2. Add JSDoc comments to provide documentation for the class and its methods.
// 3. Use a consistent naming convention for variables and methods.
// 4. Use the `Math.PI` constant instead of hardcoding the value of pi.

// Here's the refactored code:

// ```javascript
/**
 * This class represents a sphere and can calculate its volume and surface area
 * @constructor
 * @param {number} radius - The radius of the sphere
 * @see https://en.wikipedia.org/wiki/Sphere
 */
export default class Sphere {
  /**
   * Constructs a new instance of the Sphere class with the specified radius
   * @param {number} radius - The radius of the sphere
   */
  constructor(radius) {
    this.radius = radius;
  }

  /**
   * Calculates the volume of the sphere
   * @returns {number} The volume of the sphere
   */
  calculateVolume() {
    return (4 / 3) * Math.PI * Math.pow(this.radius, 3);
  }

  /**
   * Calculates the surface area of the sphere
   * @returns {number} The surface area of the sphere
   */
  calculateSurfaceArea() {
    return 4 * Math.PI * Math.pow(this.radius, 2);
  }
}
// ```

// In the refactored code, the methods `volume` and `surfaceArea` have been replaced with `calculateVolume` and `calculateSurfaceArea` respectively. Regular function syntax is used instead of arrow functions. JSDoc comments are added to provide a description and parameter types for each method. The `Math.PI` constant is used to represent the value of pi instead of hardcoding it.

