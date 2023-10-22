// To improve the maintainability of the codebase, you can make the following changes:

// 1. Use proper naming conventions for variables and functions.
// 2. Use comments to explain the purpose of the code and any complex logic.
// 3. Split the code into smaller functions to improve readability and reusability.
// 4. Add error handling and validation for the input arguments.

// Here is the refactored code:

/**
 * Encrypts a string using an XOR cipher.
 *
 * @param {string} str - The string to be encrypted.
 * @param {number} key - The key for encryption.
 * @returns {string} - The encrypted string.
 * @throws {TypeError} - If arguments are invalid.
 */
const xorCipher = (str, key) => {
  if (typeof str !== 'string' || !Number.isInteger(key)) {
    throw new TypeError('Arguments type are invalid');
  }

  return str.replace(/./g, (char) => xorCharWithKey(char, key));
};

/**
 * Applies XOR operation on a character with a key.
 *
 * @param {string} char - The character to be XORed.
 * @param {number} key - The key for encryption.
 * @returns {string} - The XORed character.
 */
const xorCharWithKey = (char, key) => {
  return String.fromCharCode(char.charCodeAt(0) ^ key);
};

export default xorCipher;

