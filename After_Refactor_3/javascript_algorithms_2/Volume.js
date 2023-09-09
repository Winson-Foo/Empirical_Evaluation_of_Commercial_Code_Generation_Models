// To improve the maintainability of the codebase, we can do the following:

// 1. Separate the calculation functions into individual files for better organization and readability.
// 2. Add proper comments to explain the purpose and usage of each function.
// 3. Use descriptive variable names instead of generic names.
// 4. Add error handling and validation to handle edge cases and invalid input.
// 5. Create a separate file for unit tests to ensure the correctness of the calculations.

// Here is the refactored code:

// cuboid.js
/*
  Calculate the volume for a Cuboid
  Reference: https://www.cuemath.com/measurement/volume-of-cuboid/
  Formula: width * length * height
*/
const calculateCuboidVolume = (width, length, height) => {
  validateNumber(width, 'Width')
  validateNumber(length, 'Length')
  validateNumber(height, 'Height')
  return width * length * height
}

export { calculateCuboidVolume }

// cube.js
/*
  Calculate the volume for a Cube
  Reference: https://www.cuemath.com/measurement/volume-of-cube/
  Formula: length * length * length
*/
const calculateCubeVolume = (length) => {
  validateNumber(length, 'Length')
  return length ** 3
}

export { calculateCubeVolume }

// cone.js
/*
  Calculate the volume for a Cone
  Reference: https://www.cuemath.com/measurement/volume-of-cone/
  Formula: PI * radius^2 * height/3
*/
const calculateConeVolume = (radius, height) => {
  validateNumber(radius, 'Radius')
  validateNumber(height, 'Height')
  return (Math.PI * radius ** 2 * height / 3.0)
}

export { calculateConeVolume }

// pyramid.js
/*
  Calculate the volume for a Pyramid
  Reference: https://www.cuemath.com/measurement/volume-of-pyramid/
  Formula: (baseLength * baseWidth * height) / 3
*/
const calculatePyramidVolume = (baseLength, baseWidth, height) => {
  validateNumber(baseLength, 'BaseLength')
  validateNumber(baseWidth, 'BaseWidth')
  validateNumber(height, 'Height')
  return (baseLength * baseWidth * height) / 3.0
}

export { calculatePyramidVolume }

// cylinder.js
/*
  Calculate the volume for a Cylinder
  Reference: https://www.cuemath.com/measurement/volume-of-cylinder/
  Formula: PI * radius^2 * height
*/
const calculateCylinderVolume = (radius, height) => {
  validateNumber(radius, 'Radius')
  validateNumber(height, 'Height')
  return (Math.PI * radius ** 2 * height)
}

export { calculateCylinderVolume }

// triangularPrism.js
/*
  Calculate the volume for a Triangular Prism
  Reference: http://lrd.kangan.edu.au/numbers/content/03_volume/04_page.htm
  Formula: 1 / 2 * baseLengthTriangle * heightTriangle * height
*/
const calculateTriangularPrismVolume = (baseLength, heightTriangle, height) => {
  validateNumber(baseLength, 'BaseLengthTriangle')
  validateNumber(heightTriangle, 'HeightTriangle')
  validateNumber(height, 'Height')
  return (1 / 2 * baseLength * heightTriangle * height)
}

export { calculateTriangularPrismVolume }

// pentagonalPrism.js
/*
  Calculate the volume for a Pentagonal Prism
  Reference: https://www.cuemath.com/measurement/volume-of-pentagonal-prism/
  Formula: 5/2 * pentagonalLength * pentagonalBaseLength * height
*/
const calculatePentagonalPrismVolume = (pentagonalLength, pentagonalBaseLength, height) => {
  validateNumber(pentagonalLength, 'PentagonalLength')
  validateNumber(pentagonalBaseLength, 'PentagonalBaseLength')
  validateNumber(height, 'Height')
  return (5 / 2 * pentagonalLength * pentagonalBaseLength * height)
}

export { calculatePentagonalPrismVolume }

// sphere.js
/*
  Calculate the volume for a Sphere
  Reference: https://www.cuemath.com/measurement/volume-of-sphere/
  Formula: 4/3 * PI * radius^3
*/
const calculateSphereVolume = (radius) => {
  validateNumber(radius, 'Radius')
  return (4 / 3 * Math.PI * radius ** 3)
}

export { calculateSphereVolume }

// hemisphere.js
/*
  Calculate the volume for a Hemisphere
  Reference: https://www.cuemath.com/measurement/volume-of-hemisphere/
  Formula: (2 * PI * radius^3)/3
*/
const calculateHemisphereVolume = (radius) => {
  validateNumber(radius, 'Radius')
  return (2.0 * Math.PI * radius ** 3) / 3.0
}

export { calculateHemisphereVolume }

// validation.js
const validateNumber = (number, name = 'number') => {
  if (typeof number !== 'number') {
    throw new TypeError(`The ${name} should be a number`)
  } else if (number < 0 || !Number.isFinite(number)) {
    throw new Error(`The ${name} should be a positive value`)
  }
}

export { validateNumber }

