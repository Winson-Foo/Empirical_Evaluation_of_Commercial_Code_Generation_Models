// To improve the maintainability of the codebase, we can follow a few steps:

// 1. Break down the code into smaller, more manageable functions.
// 2. Use meaningful variable and function names.
// 3. Remove unnecessary comments and code.
// 4. Use data structures and algorithms that are more efficient.

// Here is the refactored code:

/******************************************************
 Find and retrieve the encryption key automatically
 Note: This is a draft version, please help to modify, Thanks!
 ******************************************************/

function keyFinder(str) {
  const wordBank = [
    'I ',
    'You ',
    'We ',
    'They ',
    'He ',
    'She ',
    'It ',
    ' the ',
    'The ',
    ' of ',
    ' is ',
    'Is ',
    ' am ',
    'Am ',
    ' are ',
    'Are ',
    ' have ',
    'Have ',
    ' has ',
    'Has ',
    ' may ',
    'May ',
    ' be ',
    'Be '
  ];

  const inStr = str.toString();
  let outStr = '';

  for (let k = 0; k < 26; k++) {
    outStr = caesarCipherEncodeAndDecodeEngine(inStr, k);
    if (checkForKey(wordBank, outStr)) {
      return k;
    }
  }

  return 0;
}

function caesarCipherEncodeAndDecodeEngine(inStr, numShifted) {
  const shiftNum = numShifted;
  let outStr = '';

  for (let i = 0; i < inStr.length; i++) {
    const charCode = inStr.charCodeAt(i);
    let shiftedCharCode = charCode + shiftNum;

    if (isUpperCase(charCode)) {
      shiftedCharCode = checkUpperCaseRange(shiftedCharCode);
    } else if (isLowerCase(charCode)) {
      shiftedCharCode = checkLowerCaseRange(shiftedCharCode);
    } else if (isNumber(charCode)) {
      shiftedCharCode = checkNumberRange(shiftedCharCode);
    }

    outStr += String.fromCharCode(shiftedCharCode);
  }

  return outStr;
}

function isUpperCase(charCode) {
  return charCode >= 65 && charCode <= 90;
}

function isLowerCase(charCode) {
  return charCode >= 97 && charCode <= 122;
}

function isNumber(charCode) {
  return charCode >= 48 && charCode <= 57;
}

function checkUpperCaseRange(shiftedCharCode) {
  if (shiftedCharCode > 90) {
    const diff = (shiftedCharCode - 1 - 90) % 26;
    return 65 + diff;
  } else if (shiftedCharCode <= 64) {
    const diff = (65 - 1 - shiftedCharCode) % 26;
    return 90 - diff;
  }

  return shiftedCharCode;
}

function checkLowerCaseRange(shiftedCharCode) {
  if (shiftedCharCode > 122) {
    const diff = (shiftedCharCode - 1 - 122) % 26;
    return 97 + diff;
  } else if (shiftedCharCode <= 96) {
    const diff = (97 - 1 - shiftedCharCode) % 26;
    return 122 - diff;
  }

  return shiftedCharCode;
}

function checkNumberRange(shiftedCharCode) {
  if (shiftedCharCode > 57) {
    const diff = (shiftedCharCode - 1 - 57) % 10;
    return 48 + diff;
  } else if (shiftedCharCode < 48) {
    const diff = (48 - 1 - shiftedCharCode) % 10;
    return 57 - diff;
  }

  return shiftedCharCode;
}

function checkForKey(wordBank, outStr) {
  for (const word of wordBank) {
    if (outStr.includes(word)) {
      return true;
    }
  }
  
  return false;
}

export { keyFinder }

// > keyFinder('test')
// 0

// In the refactored code, I have:
// - Split the function `keyFinder` into smaller functions for easier understanding and testing.
// - Renamed variables and functions to be more descriptive.
// - Removed unnecessary comments.
// - Simplified the encryption and decryption logic in `caesarCipherEncodeAndDecodeEngine`.
// - Used helper functions to check the character ranges and handle wraparound.
// - Implemented a separate function `checkForKey` to check if the decrypted string contains any words from the `wordBank`.

