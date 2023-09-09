// To improve the maintainability of this codebase, you can make the following changes:

// 1. Use descriptive variable names: Instead of using generic names like `alphabet`, `shift`, and `str`, use more descriptive names that convey their purpose.

// 2. Refactor the `getCipherMap` function:
//    - Break the function into smaller, reusable functions to improve readability and maintainability.
//    - Use an object instead of an array to store the cipher map for faster lookup.
//    - Move the cyclic shift logic to a separate function for clarity.

// 3. Add meaningful comments to explain the purpose and behavior of functions and variables.

// Here's the refactored code:

// ```javascript
const englishAlphabet = 'abcdefghijklmnopqrstuvwxyz'.split('');

/**
 * Generates a cipher map out of the alphabet.
 *
 * @param {string[]} alphabet - The alphabet to generate the cipher map from.
 * @param {number} shift - The shift amount for creating the cipher map.
 * @returns {Object} - The cipher map object.
 */
const createCipherMap = (alphabet, shift) => {
  const cipherMap = {};

  // Generate the cipher map by shifting each character in the alphabet.
  alphabet.forEach((char, index) => {
    const shiftedIndex = getCyclicShiftedIndex(index, shift, alphabet.length);
    cipherMap[char] = alphabet[shiftedIndex];
  });

  return cipherMap;
};

/**
 * Performs a cyclic shift on an index.
 *
 * @param {number} index - The index to shift.
 * @param {number} shift - The shift amount.
 * @param {number} length - The length of the array.
 * @returns {number} - The cyclically shifted index.
 */
const getCyclicShiftedIndex = (index, shift, length) => {
  let shiftedIndex = (index + shift) % length;
  if (shiftedIndex < 0) {
    shiftedIndex += length;
  }
  return shiftedIndex;
};

/**
 * Encrypts a string using the Caesar cipher.
 *
 * @param {string} plainText - The string to be encrypted.
 * @param {number} shift - The shift amount for encryption.
 * @param {string[]} alphabet - The alphabet to use for encryption.
 * @returns {string} - The encrypted string.
 */
export const caesarCipherEncrypt = (plainText, shift, alphabet = englishAlphabet) => {
  const cipherMap = createCipherMap(alphabet, shift);
  const encryptedText = plainText
    .toLowerCase()
    .split('')
    .map((char) => cipherMap[char] || char)
    .join('');
  return encryptedText;
};

/**
 * Decrypts a string using the Caesar cipher.
 *
 * @param {string} cipherText - The string to be decrypted.
 * @param {number} shift - The shift amount for decryption.
 * @param {string[]} alphabet - The alphabet to use for decryption.
 * @returns {string} - The decrypted string.
 */
export const caesarCipherDecrypt = (cipherText, shift, alphabet = englishAlphabet) => {
  const cipherMap = createCipherMap(alphabet, -shift);
  const decryptedText = cipherText
    .toLowerCase()
    .split('')
    .map((char) => cipherMap[char] || char)
    .join('');
  return decryptedText;
}; 

