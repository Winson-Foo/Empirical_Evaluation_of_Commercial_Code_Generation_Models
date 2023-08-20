// To improve the maintainability of the codebase, we can make the following changes:

// 1. Remove the magic numbers and use constants instead.
// 2. Use meaningful variable names to improve code readability.
// 3. Convert the conversion functions to class methods for better code organization.
// 4. Extract out the color range validation into separate functions for reusability.
// 5. Use arrow functions for callback functions to improve code readability.

// Here is the refactored code:

// ```
/**
 * The RGB color model is an additive color model in which red, green, and blue light are added
 * together in various ways to reproduce a broad array of colors. The name of the model comes from
 * the initials of the three additive primary colors, red, green, and blue. Meanwhile, the HSV
 * representation models how colors appear under light. In it, colors are represented using three
 * components: hue, saturation and (brightness-)value. This file provides functions for converting
 * colors from one representation to the other. (description adapted from
 * https://en.wikipedia.org/wiki/RGB_color_model and https://en.wikipedia.org/wiki/HSL_and_HSV).
 */

const RGB_MAX_VALUE = 255;
const HSV_MAX_HUE = 360;
const HSV_MAX_SAT = 1;
const HSV_MAX_VAL = 1;
const HSV_HUE_EPSILON = 0.2;
const HSV_SAT_EPSILON = 0.002;
const HSV_VAL_EPSILON = 0.002;

class ColorConverter {
  /**
   * Conversion from the HSV-representation to the RGB-representation.
   *
   * @param hue Hue of the color.
   * @param saturation Saturation of the color.
   * @param value Brightness-value of the color.
   * @return The tuple of RGB-components.
   */
  static hsvToRgb(hue, saturation, value) {
    ColorConverter.validateHsvInputs(hue, saturation, value);

    const chroma = value * saturation;
    const hueSection = hue / 60;
    const secondLargestComponent = chroma * (1 - Math.abs(hueSection % 2 - 1));
    const matchValue = value - chroma;

    return ColorConverter.getRgbBySection(
      hueSection,
      chroma,
      matchValue,
      secondLargestComponent
    );
  }

  /**
   * Conversion from the RGB-representation to the HSV-representation.
   *
   * @param red Red-component of the color.
   * @param green Green-component of the color.
   * @param blue Blue-component of the color.
   * @return The tuple of HSV-components.
   */
  static rgbToHsv(red, green, blue) {
    ColorConverter.validateRgbInputs(red, green, blue);

    const dRed = red / RGB_MAX_VALUE;
    const dGreen = green / RGB_MAX_VALUE;
    const dBlue = blue / RGB_MAX_VALUE;
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

    hue = (hue + HSV_MAX_HUE) % HSV_MAX_HUE;

    return [hue, saturation, value];
  }

  static approximatelyEqualHsv(hsv1, hsv2) {
    const [h1, s1, v1] = hsv1;
    const [h2, s2, v2] = hsv2;

    const bHue = Math.abs(h1 - h2) < HSV_HUE_EPSILON;
    const bSaturation = Math.abs(s1 - s2) < HSV_SAT_EPSILON;
    const bValue = Math.abs(v1 - v2) < HSV_VAL_EPSILON;

    return bHue && bSaturation && bValue;
  }

  static getRgbBySection = (hueSection, chroma, matchValue, secondLargestComponent) => {
    const convertToInt = (input) => Math.round(RGB_MAX_VALUE * input);

    let red;
    let green;
    let blue;

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
  };

  static validateHsvInputs(hue, saturation, value) {
    ColorConverter.validateRange(hue, 0, HSV_MAX_HUE, 'hue');
    ColorConverter.validateRange(saturation, 0, HSV_MAX_SAT, 'saturation');
    ColorConverter.validateRange(value, 0, HSV_MAX_VAL, 'value');
  }

  static validateRgbInputs(red, green, blue) {
    ColorConverter.validateRange(red, 0, RGB_MAX_VALUE, 'red');
    ColorConverter.validateRange(green, 0, RGB_MAX_VALUE, 'green');
    ColorConverter.validateRange(blue, 0, RGB_MAX_VALUE, 'blue');
  }

  static validateRange(value, min, max, name) {
    if (value < min || value > max) {
      throw new Error(`${name} should be between ${min} and ${max}`);
    }
  }
}

export default ColorConverter;
// ```

// With these changes, the codebase becomes more maintainable and readable. The validation functions and constants are separated, and the conversion functions are grouped together under a class for better organization.

