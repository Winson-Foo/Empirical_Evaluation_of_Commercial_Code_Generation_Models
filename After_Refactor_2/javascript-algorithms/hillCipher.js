// To improve the maintainability of this codebase, you can make the following changes:

// 1. Use explicit imports instead of importing everything from the math/matrix/Matrix module. This will make it clear which functions or constants are being used from the module.

// ```javascript
// import { generate, dot } from '../../math/matrix/Matrix';
// ```

// 2. Rename the `generateKeyMatrix` function to `generateKeyMatrixFromKeyString` to improve clarity.

// 3. Rename the `generateMessageVector` function to `generateMessageVectorFromMessage` to improve clarity.

// 4. Split the `hillCipherEncrypt` function into smaller, more focused functions to improve readability and maintainability.

// ```javascript
/**
 * Encrypts the given message using Hill Cipher.
 *
 * @param {string} message - plaintext.
 * @param {string} keyString - key string.
 * @return {string} cipherString - encrypted message.
 */
export function hillCipherEncrypt(message, keyString) {
  // The keyString and message can only contain letters.
  const onlyLettersRegExp = /^[a-zA-Z]+$/;
  if (!onlyLettersRegExp.test(message) || !onlyLettersRegExp.test(keyString)) {
    throw new Error('The message and key string can only contain letters');
  }

  const keyMatrix = generateKeyMatrixFromKeyString(keyString);
  const messageVector = generateMessageVectorFromMessage(message);

  validateKeyMatrixLength(keyMatrix, message);

  const cipherVector = performDotProduct(keyMatrix, messageVector);
  const cipherString = convertCipherVectorToString(cipherVector);

  return cipherString;
}

/**
 * Generates key matrix from given keyString.
 *
 * @param {string} keyString - a string to build a key matrix (must be of matrixSize^2 length).
 * @return {number[][]} keyMatrix
 */
const generateKeyMatrixFromKeyString = (keyString) => {
  const matrixSize = Math.sqrt(keyString.length);
  if (!Number.isInteger(matrixSize)) {
    throw new Error(
      'Invalid key string length. The square root of the key string must be an integer',
    );
  }
  let keyStringIndex = 0;
  return generate(
    [matrixSize, matrixSize],
    // Callback to get a value of each matrix cell.
    // The order the matrix is being filled in is from left to right, from top to bottom.
    () => {
      // A → 0, B → 1, ..., a → 32, b → 33, ...
      const charCodeShifted = (keyString.codePointAt(keyStringIndex)) % alphabetCodeShift;
      keyStringIndex += 1;
      return charCodeShifted;
    },
  );
};

/**
 * Generates a message vector from a given message.
 *
 * @param {string} message - the message to encrypt.
 * @return {number[][]} messageVector
 */
const generateMessageVectorFromMessage = (message) => {
  return generate(
    [message.length, 1],
    // Callback to get a value of each matrix cell.
    // The order the matrix is being filled in is from left to right, from top to bottom.
    (cellIndices) => {
      const rowIndex = cellIndices[0];
      return message.codePointAt(rowIndex) % alphabetCodeShift;
    },
  );
};

/**
 * Validates the length of the key matrix.
 *
 * @param {number[][]} keyMatrix - key matrix.
 * @param {string} message - plaintext message.
 */
const validateKeyMatrixLength = (keyMatrix, message) => {
  if (keyMatrix.length !== message.length) {
    throw new Error('Invalid key string length. The key length must be a square of message length');
  }
};

/**
 * Performs dot product of key matrix and message vector.
 *
 * @param {number[][]} keyMatrix - key matrix.
 * @param {number[][]} messageVector - message vector.
 * @return {number[][]} cipherVector - result of dot product.
 */
const performDotProduct = (keyMatrix, messageVector) => {
  return dot(keyMatrix, messageVector);
};

/**
 * Converts the cipher vector to a string.
 *
 * @param {number[][]} cipherVector - cipher vector.
 * @return {string} cipherString - encrypted message.
 */
const convertCipherVectorToString = (cipherVector) => {
  let cipherString = '';
  for (let row = 0; row < cipherVector.length; row += 1) {
    const item = cipherVector[row];
    cipherString += String.fromCharCode((item % englishAlphabetSize) + alphabetCodeShift);
  }
  return cipherString;
};
// ```

// 5. Provide an implementation for the `hillCipherDecrypt` function.

