// To improve the maintainability of the codebase, here are some suggested refactorings:

// 1. Split the `hash` function into smaller, more focused functions. This will improve readability and make the code easier to understand and maintain.

// 2. Replace some magic numbers with named constants to improve code readability and maintainability.

// 3. Use arrow function expressions for shorter, more concise code.

// 4. Add type annotations/comments to clarify the purpose and expected types of the parameters and return values.

// Here is the refactored code:

// ```javascript
const DEFAULT_BASE = 37;
const DEFAULT_MODULUS = 101;

export default class PolynomialHash {
  /**
   * @param {number} [base] - Base number that is used to create the polynomial.
   * @param {number} [modulus] - Modulus number that keeps the hash from overflowing.
   */
  constructor({ base = DEFAULT_BASE, modulus = DEFAULT_MODULUS } = {}) {
    this.base = base;
    this.modulus = modulus;
  }

  /**
   * Function that creates hash representation of the word.
   *
   * Time complexity: O(word.length).
   *
   * @param {string} word - String that needs to be hashed.
   * @return {number} - The hash value of the word.
   */
  hash(word) {
    const charCodes = Array.from(word).map(this.charToNumber);

    let hash = 0;
    charCodes.forEach((charCode) => {
      hash = this.updateHash(hash, charCode);
    });

    return hash;
  }

  /**
   * Function that creates hash representation of the word
   * based on previous word (shifted by one character left) hash value.
   *
   * Recalculates the hash representation of a word so that it isn't
   * necessary to traverse the whole word again.
   *
   * Time complexity: O(1).
   *
   * @param {number} prevHash - The hash value of the previous word.
   * @param {string} prevWord - The previous word.
   * @param {string} newWord - The new word.
   * @return {number} - The hash value of the new word.
   */
  roll(prevHash, prevWord, newWord) {
    const prevValue = this.charToNumber(prevWord[0]);
    const newValue = this.charToNumber(newWord[newWord.length - 1]);

    const prevValueMultiplier = this.calculateMultiplier(prevWord.length - 1);

    let hash = this.subtractProduct(prevHash, prevValue, prevValueMultiplier);
    hash = this.updateHash(hash, newValue);

    return hash;
  }

  /**
   * Converts character to number.
   *
   * @param {string} char - The character to convert.
   * @return {number} - The number representation of the character.
   */
  charToNumber = (char) => {
    let charCode = char.codePointAt(0);

    // Check if character has surrogate pair.
    const surrogate = char.codePointAt(1);
    if (surrogate !== undefined) {
      const surrogateShift = 2 ** 16;
      charCode += surrogate * surrogateShift;
    }

    return charCode;
  };

  /**
   * Updates the hash value with the new character code.
   *
   * @param {number} hash - The current hash value.
   * @param {number} charCode - The character code to include in the hash.
   * @return {number} - The updated hash value.
   */
  updateHash(hash, charCode) {
    hash *= this.base;
    hash += charCode;
    hash %= this.modulus;

    return hash;
  }

  /**
   * Calculates the multiplier for the previous value.
   *
   * @param {number} exponent - The exponent value.
   * @return {number} - The calculated multiplier.
   */
  calculateMultiplier(exponent) {
    let multiplier = 1;
    for (let i = 0; i < exponent; i += 1) {
      multiplier = (multiplier * this.base) % this.modulus;
    }

    return multiplier;
  }

  /**
   * Subtracts the product of the two values, taking modulus into account.
   *
   * @param {number} value1 - The first value.
   * @param {number} value2 - The second value.
   * @param {number} multiplier - The multiplier used in the product calculation.
   * @return {number} - The subtracted product value, taking modulus into account.
   */
  subtractProduct(value1, value2, multiplier) {
    let product = (value2 * multiplier) % this.modulus;
    value1 -= product;

    if (value1 < 0) {
      value1 += this.modulus;
    }

    return value1;
  }
}
// ```

// These refactorings improve the maintainability of the code by breaking it down into smaller, focused functions, using named constants, and adding type annotations/comments to clarify the purpose and expected types of the parameters and return values.

