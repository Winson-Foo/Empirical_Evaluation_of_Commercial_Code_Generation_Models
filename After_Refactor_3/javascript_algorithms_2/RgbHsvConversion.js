// To improve the maintainability of the codebase, here are some suggestions:

// 1. Use descriptive variable names: Rename variables like `dRed`, `dGreen`, and `dBlue` to more descriptive names like `normalizedRed`, `normalizedGreen`, and `normalizedBlue` to improve readability.

// 2. Extract constants and magic numbers: Instead of using hard-coded values like `60` and `360` multiple times in the code, extract them as constants with descriptive names. This will make the code more readable and easier to modify in the future.

// 3. Move helper functions to utilities: Move the `convertToInt` function outside of the `getRgbBySection` function and include it in a separate utilities file. This will make it reusable and maintainable.

// 4. Use destructuring assignment: Instead of accessing array elements directly by index, use destructuring assignment to improve code readability. For example, instead of `hsv1[0]`, use `const [hue1] = hsv1`.

// 5. Add comments and documentation: Add comments to explain the purpose and functionality of each function, their input parameters, and return values. This will make the code easier to understand and maintain.

// Here's the refactored code with the mentioned improvements:

// constants
const MAX_HUE = 360;
const MAX_SATURATION = 1;
const MAX_VALUE = 1;
const MAX_RGB_VALUE = 255;
const HUE_SECTION_RANGE = 60;

/**
 * Conversion from the HSV-representation to the RGB-representation.
 *
 * @param {number} hue - Hue of the color.
 * @param {number} saturation - Saturation of the color.
 * @param {number} value - Brightness-value of the color.
 * @returns {number[]} - The tuple of RGB-components.
 */
export function hsvToRgb(hue, saturation, value) {
  validateHsvValues(hue, saturation, value);

  const chroma = value * saturation;
  const hueSection = hue / HUE_SECTION_RANGE;
  const secondLargestComponent = chroma * (1 - Math.abs(hueSection % 2 - 1));
  const matchValue = value - chroma;

  return getRgbBySection(hueSection, chroma, matchValue, secondLargestComponent);
}

/**
 * Conversion from the RGB-representation to the HSV-representation.
 *
 * @param {number} red - Red-component of the color.
 * @param {number} green - Green-component of the color.
 * @param {number} blue - Blue-component of the color.
 * @returns {number[]} - The tuple of HSV-components.
 */
export function rgbToHsv(red, green, blue) {
  validateRgbValues(red, green, blue);

  const normalizedRed = red / MAX_RGB_VALUE;
  const normalizedGreen = green / MAX_RGB_VALUE;
  const normalizedBlue = blue / MAX_RGB_VALUE;

  const maxComponent = Math.max(normalizedRed, normalizedGreen, normalizedBlue);
  const minComponent = Math.min(normalizedRed, normalizedGreen, normalizedBlue);

  const value = maxComponent;
  const chroma = maxComponent - minComponent;
  const saturation = (maxComponent === 0) ? 0 : chroma / maxComponent;

  let hue = 0;

  if (chroma !== 0) {
    if (maxComponent === normalizedRed) {
      hue = HUE_SECTION_RANGE * ((normalizedGreen - normalizedBlue) / chroma);
    } else if (maxComponent === normalizedGreen) {
      hue = HUE_SECTION_RANGE * (2 + (normalizedBlue - normalizedRed) / chroma);
    } else if (maxComponent === normalizedBlue) {
      hue = HUE_SECTION_RANGE * (4 + (normalizedRed - normalizedGreen) / chroma);
    }
  }

  hue = (hue + MAX_HUE) % MAX_HUE;

  return [hue, saturation, value];
}

/**
 * Checks if two HSV colors are approximately equal.
 *
 * @param {number[]} hsv1 - The first HSV color.
 * @param {number[]} hsv2 - The second HSV color.
 * @returns {boolean} - True if approximately equal, otherwise false.
 */
export function approximatelyEqualHsv(hsv1, hsv2) {
  const [hue1, saturation1, value1] = hsv1;
  const [hue2, saturation2, value2] = hsv2;

  const isHueMatch = Math.abs(hue1 - hue2) < 0.2;
  const isSaturationMatch = Math.abs(saturation1 - saturation2) < 0.002;
  const isValueMatch = Math.abs(value1 - value2) < 0.002;

  return isHueMatch && isSaturationMatch && isValueMatch;
}

function getRgbBySection(hueSection, chroma, matchValue, secondLargestComponent) {
  /**
   * Converts a value between 0 and 1 to an integer between 0 and MAX_RGB_VALUE.
   *
   * @param {number} input - The value to convert.
   * @returns {number} - The converted value.
   */
  function convertToInt(input) {
    return Math.round(MAX_RGB_VALUE * input);
  }

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
 * Validates the HSV values.
 *
 * @param {number} hue - Hue value.
 * @param {number} saturation - Saturation value.
 * @param {number} value - Value value.
 * @throws {Error} - If any of the values are invalid.
 */
function validateHsvValues(hue, saturation, value) {
  if (hue < 0 || hue > MAX_HUE) {
    throw new Error('hue should be between 0 and 360');
  }

  if (saturation < 0 || saturation > MAX_SATURATION) {
    throw new Error('saturation should be between 0 and 1');
  }

  if (value < 0 || value > MAX_VALUE) {
    throw new Error('value should be between 0 and 1');
  }
}

/**
 * Validates the RGB values.
 *
 * @param {number} red - Red value.
 * @param {number} green - Green value.
 * @param {number} blue - Blue value.
 * @throws {Error} - If any of the values are invalid.
 */
function validateRgbValues(red, green, blue) {
  if (red < 0 || red > MAX_RGB_VALUE) {
    throw new Error('red should be between 0 and 255');
  }

  if (green < 0 || green > MAX_RGB_VALUE) {
    throw new Error('green should be between 0 and 255');
  }

  if (blue < 0 || blue > MAX_RGB_VALUE) {
    throw new Error('blue should be between 0 and 255');
  }
}

