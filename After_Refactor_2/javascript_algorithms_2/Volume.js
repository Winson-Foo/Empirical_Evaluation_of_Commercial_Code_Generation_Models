// To improve the maintainability of the codebase, you can consider the following refactored code:

// ```javascript
// Shapes.js

class Shape {
  constructor() {
    if (new.target === Shape) {
      throw new TypeError("Cannot instantiate abstract class Shape.");
    }
  }

  calculateVolume() {
    throw new Error("Method calculateVolume() must be implemented.");
  }

  validateNumber(number, name = 'number') {
    if (typeof number !== 'number') {
      throw new TypeError(`The ${name} should be of type Number.`);
    } else if (number < 0 || !Number.isFinite(number)) {
      throw new Error(`The ${name} only accepts positive values.`);
    }
  }
}

class Cuboid extends Shape {
  constructor(width, length, height) {
    super();
    this.width = width;
    this.length = length;
    this.height = height;
  }

  calculateVolume() {
    super.validateNumber(this.width, 'Width');
    super.validateNumber(this.length, 'Length');
    super.validateNumber(this.height, 'Height');
    return this.width * this.length * this.height;
  }
}

class Cube extends Shape {
  constructor(length) {
    super();
    this.length = length;
  }

  calculateVolume() {
    super.validateNumber(this.length, 'Length');
    return Math.pow(this.length, 3);
  }
}

class Cone extends Shape {
  constructor(radius, height) {
    super();
    this.radius = radius;
    this.height = height;
  }

  calculateVolume() {
    super.validateNumber(this.radius, 'Radius');
    super.validateNumber(this.height, 'Height');
    return (Math.PI * Math.pow(this.radius, 2) * this.height) / 3.0;
  }
}

class Pyramid extends Shape {
  constructor(baseLength, baseWidth, height) {
    super();
    this.baseLength = baseLength;
    this.baseWidth = baseWidth;
    this.height = height;
  }

  calculateVolume() {
    super.validateNumber(this.baseLength, 'BaseLength');
    super.validateNumber(this.baseWidth, 'BaseWidth');
    super.validateNumber(this.height, 'Height');
    return (this.baseLength * this.baseWidth * this.height) / 3.0;
  }
}

class Cylinder extends Shape {
  constructor(radius, height) {
    super();
    this.radius = radius;
    this.height = height;
  }

  calculateVolume() {
    super.validateNumber(this.radius, 'Radius');
    super.validateNumber(this.height, 'Height');
    return Math.PI * Math.pow(this.radius, 2) * this.height;
  }
}

class TriangularPrism extends Shape {
  constructor(baseLength, heightTriangle, height) {
    super();
    this.baseLength = baseLength;
    this.heightTriangle = heightTriangle;
    this.height = height;
  }

  calculateVolume() {
    super.validateNumber(this.baseLength, 'BaseLengthTriangle');
    super.validateNumber(this.heightTriangle, 'HeightTriangle');
    super.validateNumber(this.height, 'Height');
    return (1 / 2 * this.baseLength * this.heightTriangle * this.height);
  }
}

class PentagonalPrism extends Shape {
  constructor(pentagonalLength, pentagonalBaseLength, height) {
    super();
    this.pentagonalLength = pentagonalLength;
    this.pentagonalBaseLength = pentagonalBaseLength;
    this.height = height;
  }

  calculateVolume() {
    super.validateNumber(this.pentagonalLength, 'PentagonalLength');
    super.validateNumber(this.pentagonalBaseLength, 'PentagonalBaseLength');
    super.validateNumber(this.height, 'Height');
    return (5 / 2 * this.pentagonalLength * this.pentagonalBaseLength * this.height);
  }
}

class Sphere extends Shape {
  constructor(radius) {
    super();
    this.radius = radius;
  }

  calculateVolume() {
    super.validateNumber(this.radius, 'Radius');
    return (4 / 3 * Math.PI * Math.pow(this.radius, 3));
  }
}

class Hemisphere extends Shape {
  constructor(radius) {
    super();
    this.radius = radius;
  }

  calculateVolume() {
    super.validateNumber(this.radius, 'Radius');
    return (2.0 * Math.PI * Math.pow(this.radius, 3)) / 3.0;
  }
}

export {
  Cuboid,
  Cube,
  Cone,
  Pyramid,
  Cylinder,
  TriangularPrism,
  PentagonalPrism,
  Sphere,
  Hemisphere
};
// ```

// With the refactored code, the volume calculations for each shape are organized into separate classes that inherit from the abstract class `Shape`. Each shape class contains a `calculateVolume()` method that calculates the volume based on the specified dimensions.

// The `validateNumber()` method is moved to the `Shape` class as a common validation method, and the specific shape classes call this method to validate their input dimensions.

// By using classes and separating the functionality into smaller, more manageable objects, the codebase becomes more modular and maintainable.

