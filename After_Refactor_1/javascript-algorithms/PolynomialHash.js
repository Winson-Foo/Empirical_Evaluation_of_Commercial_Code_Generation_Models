// To improve the maintainability of the codebase, we can make the following changes:

// 1. Extract the calculations related to `prevValueMultiplier` into a separate function to improve readability.
// 2. Remove unnecessary comments that repeat information or do not provide additional value.
// 3. Simplify the logic for handling surrogate pairs.

// Here is the refactored code:

// ```javascript
const DEFAULT_BASE = 37;
const DEFAULT_MODULUS = 101;

export default class PolynomialHash {
  constructor({ base = DEFAULT_BASE, modulus = DEFAULT_MODULUS } = {}) {
    this.base = base;
    this.modulus = modulus;
  }

  hash(word) {
    const charCodes = Array.from(word).map((char) => this.charToNumber(char));
    let hash = 0;

    for (const charCode of charCodes) {
      hash *= this.base;
      hash += charCode;
      hash %= this.modulus;
    }

    return hash;
  }

  roll(prevHash, prevWord, newWord) {
    let hash = prevHash;

    const prevValue = this.charToNumber(prevWord[0]);
    const newValue = this.charToNumber(newWord[newWord.length - 1]);
    const prevValueMultiplier = this.getPrevValueMultiplier(prevWord.length - 1);

    hash += this.modulus;
    hash -= (prevValue * prevValueMultiplier) % this.modulus;

    hash *= this.base;
    hash += newValue;
    hash %= this.modulus;

    return hash;
  }

  charToNumber(char) {
    let charCode = char.codePointAt(0);

    // Handle surrogate pairs
    const surrogate = char.codePointAt(1);
    if (surrogate !== undefined) {
      charCode = (charCode << 16) + surrogate;
    }

    return charCode;
  }

  getPrevValueMultiplier(power) {
    let multiplier = 1;

    for (let i = 1; i <= power; i++) {
      multiplier *= this.base;
      multiplier %= this.modulus;
    }

    return multiplier;
  }
}
// ```

// These changes improve the maintainability of the codebase by making the code more readable and removing unnecessary complexity.

