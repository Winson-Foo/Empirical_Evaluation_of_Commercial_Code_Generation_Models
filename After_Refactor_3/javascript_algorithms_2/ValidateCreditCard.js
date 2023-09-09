// To improve the maintainability of the codebase, here are some suggested changes:

// 1. Use meaningful variable and function names: Rename variables and functions to more descriptive names that reflect their purpose and functionality. This improves code readability and makes it easier to understand.

// 2. Split complex logic into separate functions: Split the logic into smaller, more manageable functions. This promotes code reusability and makes it easier to test and maintain.

// 3. Use constants for error messages and valid start substrings: Declare constants for error messages and valid start substrings to avoid repeating them in multiple places. This helps in maintaining consistency and makes it easier to update or modify them in the future.

// 4. Use early returns instead of long if-else chains: Replace long if-else chains with early returns when possible. This simplifies the code and reduces the nesting depth, making it easier to follow the control flow.

// Here's the refactored code:

const VALID_START_SUBSTRINGS = ['4', '5', '6', '37', '34', '35']; // Valid credit card numbers start with these numbers
const INVALID_CARD_TYPE_ERROR = 'it has nonnumerical characters.';
const INVALID_CARD_LENGTH_ERROR = 'of its length.';
const INVALID_CARD_START_ERROR = 'of its first two digits.';
const LUHN_CHECK_ERROR = 'it fails the Luhn check.';

const luhnValidation = (creditCardNumber) => {
  let validationSum = 0;

  creditCardNumber.split('').forEach((digit, index) => {
    let currentDigit = parseInt(digit);
    
    if (index % 2 === 0) {
      currentDigit *= 2;

      if (currentDigit > 9) {
        currentDigit %= 10;
        currentDigit += 1;
      }
    }
    
    validationSum += currentDigit;
  });

  return validationSum % 10 === 0;
};

const validateCreditCard = (creditCardString) => {
  if (typeof creditCardString !== 'string') {
    throw new TypeError('The given value is not a string');
  }

  const errorMessage = `${creditCardString} is an invalid credit card number because `;

  if (isNaN(creditCardString)) {
    throw new TypeError(errorMessage + INVALID_CARD_TYPE_ERROR);
  }

  const creditCardStringLength = creditCardString.length;

  if (!((creditCardStringLength >= 13) && (creditCardStringLength <= 16))) {
    throw new Error(errorMessage + INVALID_CARD_LENGTH_ERROR);
  }

  if (!validStartSubString(creditCardString)) {
    throw new Error(errorMessage + INVALID_CARD_START_ERROR);
  }

  if (!luhnValidation(creditCardString)) {
    throw new Error(errorMessage + LUHN_CHECK_ERROR);
  }

  return true;
};

const validStartSubString = (creditCardString) => {
  return VALID_START_SUBSTRINGS.some(subString => creditCardString.startsWith(subString));
};

export { validateCreditCard };

