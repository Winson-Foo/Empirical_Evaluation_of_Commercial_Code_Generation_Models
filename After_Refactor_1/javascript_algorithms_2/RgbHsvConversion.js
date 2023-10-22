// To improve the maintainability of this codebase, we can refactor the code by introducing helper functions, removing duplicated code, and adding documentation. Here is the refactored code:

// ```javascript
/**
 * The RGB color model is an additive color model in which red, green, and blue light are added
 * together in various ways to reproduce a broad array of colors. The name of the model comes from
 * the initials of the three additive primary colors, red, green, and blue. Meanwhile, the HSV
 * representation models how colors appear under light. In it, colors are represented using three
 * components: hue, saturation and (brightness-)value.
 *
 * This file provides functions for converting colors from one representation to the other.
 * (description adapted from https://en.wikipedia.org/wiki/RGB_color_model and
 * https://en.wikipedia.org/wiki/HSL_and_HSV).
 */

/**
 * Conversion from the HSV-representation to the RGB-representation.
 *
 * @param {number} hue - Hue of the color (between 0 and 360).
 * @param {number} saturation - Saturation of the color (between 0 and 1).
 * @param {number} value - Brightness-value of the color (between 0 and 1).
 * @returns {number[]} - The tuple of RGB-components.
 * @throws {Error} - Throws error if hue, saturation, or value is out of valid range.
 */
export function hsvToRgb(hue, saturation, value) {
  validateHsvInputs(hue, saturation, value);

  const chroma = value * saturation;
  const hueSection = hue / 60;
  const secondLargestComponent = chroma * (1 - Math.abs(hueSection % 2 - 1));
  const matchValue = value - chroma;

  return getRgbBySection(hueSection, chroma, matchValue, secondLargestComponent);
}

/**
 * Conversion from the RGB-representation to the HSV-representation.
 *
 * @param {number} red - Red-component of the color (between 0 and 255).
 * @param {number} green - Green-component of the color (between 0 and 255).
 * @param {number} blue - Blue-component of the color (between 0 and 255).
 * @returns {number[]} - The tuple of HSV-components.
 * @throws {Error} - Throws error if red, green, or blue is out of valid range.
 */
export function rgbToHsv(red, green, blue) {
  validateRgbInputs(red, green, blue);

  const dRed = red / 255;
  const dGreen = green / 255;
  const dBlue = blue / 255;
  const value = Math.max(Math.max(dRed, dGreen), dBlue);
  const chroma = value - Math.min(Math.min(dRed, dGreen), dBlue);
  const saturation = value === 0 ? 0 : chroma / value;

  let hue;
  if (chroma === 0) {
    hue = 0;
  } else if (value === dRed) {
    hue = 60 * ((dGreen - dBlue) / chroma);
  } else if (value === dGreen) {
    hue = 60 * (2 + (dBlue - dRed) / chroma);
  } else {
    hue = 60 * (4 + (dRed - dGreen) / chroma);
  }

  hue = (hue + 360) % 360;

  return [hue, saturation, value];
}

/**
 * Checks if two HSV colors are approximately equal with a certain tolerance.
 *
 * @param {number[]} hsv1 - The first HSV color as a tuple.
 * @param {number[]} hsv2 - The second HSV color as a tuple.
 * @returns {boolean} - Returns true if the two colors are approximately equal.
 */
export function approximatelyEqualHsv(hsv1, hsv2) {
  const bHue = Math.abs(hsv1[0] - hsv2[0]) < 0.2;
  const bSaturation = Math.abs(hsv1[1] - hsv2[1]) < 0.002;
  const bValue = Math.abs(hsv1[2] - hsv2[2]) < 0.002;

  return bHue && bSaturation && bValue;
}

/**
 * Converts a floating-point color component to an integer between 0 and 255.
 *
 * @param {number} input - The floating-point color component (between 0 and 1).
 * @returns {number} - The corresponding integer color component between 0 and 255.
 */
function convertToInt(input) {
  return Math.round(255 * input);
}

/**
 * Get the RGB components based on the hue section.
 *
 * @param {number} hueSection - The hue section (between 0 and 6).
 * @param {number} chroma - The chroma component.
 * @param {number} matchValue - The match value.
 * @param {number} secondLargestComponent - The second largest component.
 * @returns {number[]} - The RGB components.
 */
function getRgbBySection(hueSection, chroma, matchValue, secondLargestComponent) {
  let red, green, blue;

  if (hueSection >= 0 && hueSection <= 1) {
    red = convertToInt(chroma + matchValue);
    green = convertToInt(secondLargestComponent + matchValue);
    blue = convertToInt(matchValue);
  } else if (hueSection > 1 && hueSection <= 2) {
    red = convertToInt(secondLargestComponent + matchValue);
    green = convertToInt(chroma + matchValue);
    blue = convertToInt(matchValue);
  } else if (hueSection > 2 && hueSection <= 3) {
    red = convertToInt(matchValue);
    green = convertToInt(chroma + matchValue);
    blue = convertToInt(secondLargestComponent + matchValue);
  } else if (hueSection > 3 && hueSection <= 4) {
    red = convertToInt(matchValue);
    green = convertToInt(secondLargestComponent + matchValue);
    blue = convertToInt(chroma + matchValue);
  } else if (hueSection > 4 && hueSection <= 5) {
    red = convertToInt(secondLargestComponent + matchValue);
    green = convertToInt(matchValue);
    blue = convertToInt(chroma + matchValue);
  } else {
    red = convertToInt(chroma + matchValue);
    green = convertToInt(matchValue);
    blue = convertToInt(secondLargestComponent + matchValue);
  }

  return [red, green, blue];
}

/**
 * Validates the HSV inputs to ensure they are within valid range.
 *
 * @param {number} hue - The hue value.
 * @param {number} saturation - The saturation value.
 * @param {number} value - The value value.
 * @throws {Error} - Throws an error if the inputs are out of valid range.
 */
function validateHsvInputs(hue, saturation, value) {
  if (hue < 0 || hue > 360) {
    throw new Error('hue should be between 0 and 360');
  }

  if (saturation < 0 || saturation > 1) {
    throw new Error('saturation should be between 0 and 1');
  }

  if (value < 0 || value > 1) {
    throw new Error('value should be between 0 and 1');
  }
}

/**
 * Validates the RGB inputs to ensure they are within valid range.
 *
 * @param {number} red - The red value.
 * @param {number} green - The green value.
 * @param {number} blue - The blue value.
 * @throws {Error} - Throws an error if the inputs are out of valid range.
 */
function validateRgbInputs(red, green, blue) {
  if (red < 0 || red > 255) {
    throw new Error('red should be between 0 and 255');
  }

  if (green < 0 || green > 255) {
    throw new Error('green should be between 0 and 255');
  }

  if (blue < 0 || blue > 255) {
    throw new Error('blue should be between 0 and 255');
  }
}
// ```

// The refactored code separates the validation of inputs into separate validation functions and adds appropriate comments and JSDoc annotations to improve the code's readability and maintainability.

