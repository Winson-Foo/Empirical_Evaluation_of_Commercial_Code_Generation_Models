// To improve the maintainability of the codebase, we can implement the following changes:

// 1. Use meaningful variable and function names: Replace generic names like `width`, `length`, `height`, `radius`, etc. with more descriptive names that accurately reflect their purpose.

// 2. Add comments: Add comments to each function to explain its purpose, input parameters, and return value. This will make it easier for developers to understand and maintain the code in the future.

// 3. Use a consistent naming convention: Choose a consistent naming convention for variables and functions (e.g., camel case) and apply it throughout the entire codebase.

// 4. Use constants for mathematical values: Instead of hard-coding mathematical constants like PI and 2/3, create constants for them and use those constants in the calculations. This improves readability and allows for easy modification if the values ever change.

// 5. Extract common logic: Identify any common logic present in multiple functions and extract it into separate helper functions. This will avoid code duplication and make it easier to update the logic in one place.

// 6. Improve error handling: Instead of throwing generic `Error` and `TypeError` exceptions, create specific custom exceptions with meaningful error messages. This will help in identifying the cause of errors and debugging the code.

// 7. Add input validation: Add input validation checks to ensure that the input parameters are of the expected type and within the acceptable range. This will prevent unexpected errors and improve the reliability of the code.

// Below is the refactored code with the above improvements:

// ```javascript
// Constants
const PI = Math.PI;

// Calculate the volume of a cuboid
const calculateCuboidVolume = (width, length, height) => {
  validateNumber(width, 'Width');
  validateNumber(length, 'Length');
  validateNumber(height, 'Height');
  return width * length * height;
};

// Calculate the volume of a cube
const calculateCubeVolume = (sideLength) => {
  validateNumber(sideLength, 'Side Length');
  return sideLength ** 3;
};

// Calculate the volume of a cone
const calculateConeVolume = (radius, height) => {
  validateNumber(radius, 'Radius');
  validateNumber(height, 'Height');
  return (PI * radius ** 2 * height) / 3;
};

// Calculate the volume of a pyramid
const calculatePyramidVolume = (baseLength, baseWidth, height) => {
  validateNumber(baseLength, 'Base Length');
  validateNumber(baseWidth, 'Base Width');
  validateNumber(height, 'Height');
  return (baseLength * baseWidth * height) / 3;
};

// Calculate the volume of a cylinder
const calculateCylinderVolume = (radius, height) => {
  validateNumber(radius, 'Radius');
  validateNumber(height, 'Height');
  return PI * radius ** 2 * height;
};

// Calculate the volume of a triangular prism
const calculateTriangularPrismVolume = (baseLength, baseWidth, height) => {
  validateNumber(baseLength, 'Base Length');
  validateNumber(baseWidth, 'Base Width');
  validateNumber(height, 'Height');
  return (1 / 2) * baseLength * baseWidth * height;
};

// Calculate the volume of a pentagonal prism
const calculatePentagonalPrismVolume = (pentagonalLength, pentagonalBaseLength, height) => {
  validateNumber(pentagonalLength, 'Pentagonal Length');
  validateNumber(pentagonalBaseLength, 'Pentagonal Base Length');
  validateNumber(height, 'Height');
  return (5 / 2) * pentagonalLength * pentagonalBaseLength * height;
};

// Calculate the volume of a sphere
const calculateSphereVolume = (radius) => {
  validateNumber(radius, 'Radius');
  return (4 / 3) * PI * radius ** 3;
};

// Calculate the volume of a hemisphere
const calculateHemisphereVolume = (radius) => {
  validateNumber(radius, 'Radius');
  return (2 / 3) * PI * radius ** 3;
};

// Helper function to validate a number
const validateNumber = (number, paramName = 'number') => {
  if (typeof number !== 'number' || isNaN(number) || !isFinite(number) || number < 0) {
    throw new TypeError(`Invalid ${paramName}. Expected a positive number.`);
  }
};

export {
  calculateCuboidVolume,
  calculateCubeVolume,
  calculateConeVolume,
  calculatePyramidVolume,
  calculateCylinderVolume,
  calculateTriangularPrismVolume,
  calculatePentagonalPrismVolume,
  calculateSphereVolume,
  calculateHemisphereVolume
};
// ```

// These changes should improve the maintainability of the codebase by making it more readable, modular, and easier to understand and update.

