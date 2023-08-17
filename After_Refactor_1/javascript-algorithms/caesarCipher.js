// To improve the maintainability of the codebase, we can make the following refactorings:

// 1. Move the logic to create the cipher map into a separate function.
// 2. Use more descriptive variable names to improve code readability.
// 3. Use a separate function to convert the string to lowercase and split it into characters.
// 4. Use a separate function to map each character using the cipher map and join them back into a string.

// Here is the refactored code:

// ```javascript
const englishAlphabet = 'abcdefghijklmnopqrstuvwxyz'.split('');

/**
 * Generates a cipher map out of the alphabet.
 * Example with a shift 3: {'a': 'd', 'b': 'e', 'c': 'f', ...}
 *
 * @param {string[]} alphabet - i.e. ['a', 'b', 'c', ... , 'z']
 * @param {number} shift - i.e. 3
 * @return {Object} - i.e. {'a': 'd', 'b': 'e', 'c': 'f', ..., 'z': 'c'}
 */
const generateCipherMap = (alphabet, shift) => {
  return alphabet.reduce((cipherMap, currentChar, charIndex) => {
    const shiftedIndex = (charIndex + shift) % alphabet.length;
    const encryptedCharIndex = shiftedIndex < 0 ? shiftedIndex + alphabet.length : shiftedIndex;
    cipherMap[currentChar] = alphabet[encryptedCharIndex];
    return cipherMap;
  }, {});
};

/**
 * @param {string} str
 * @return {string[]}
 */
const convertStringToCharacters = (str) => {
  return str.toLowerCase().split('');
};

/**
 * @param {string[]} characters
 * @param {Object} cipherMap
 * @return {string}
 */
const mapCharacters = (characters, cipherMap) => {
  return characters.map((char) => cipherMap[char] || char).join('');
};

/**
 * @param {string} str
 * @param {number} shift
 * @param {string[]} alphabet
 * @return {string}
 */
export const caesarCipherEncrypt = (str, shift, alphabet = englishAlphabet) => {
  const cipherMap = generateCipherMap(alphabet, shift);
  const characters = convertStringToCharacters(str);
  return mapCharacters(characters, cipherMap);
};

/**
 * @param {string} str
 * @param {number} shift
 * @param {string[]} alphabet
 * @return {string}
 */
export const caesarCipherDecrypt = (str, shift, alphabet = englishAlphabet) => {
  const cipherMap = generateCipherMap(alphabet, -shift);
  const characters = convertStringToCharacters(str);
  return mapCharacters(characters, cipherMap);
};
// ```

// By separating the code into smaller, more focused functions, it becomes easier to understand, test, and maintain.

