// To improve the maintainability of this codebase, here are some refactorings:

// 1. Extract functions to improve readability and reusability:
//    - Extract a function `isNumeric` to check if a value is a number.
//    - Extract a function `hasValidLength` to check if a credit card number has a valid length.
//    - Extract a function `startsWithValidSubString` to check if a credit card number starts with a valid substring.
//    - Extract the Luhn algorithm logic into a separate function `calculateLuhnValidationSum`.

// 2. Use early returns instead of nested conditional statements for error handling.

// 3. Use more descriptive variable names.

// 4. Add comments to explain the purpose and logic of each section of code.

// Here's the refactored code:

// ```javascript
/**
 * Validate a given credit card number
 *
 * The core of the validation of credit card numbers is the Luhn algorithm.
 *
 * The validation sum should be completely divisible by 10 which is calculated as follows,
 * every first digit is added directly to the validation sum.
 * For every second digit in the credit card number, the digit is multiplied by 2.
 * If the product is greater than 10 the digits of the product are added.
 * This resultant digit is considered for the validation sum rather than the digit itself.
 *
 * Ref: https://www.geeksforgeeks.org/luhn-algorithm/
 */

// Helper function to check if a value is a number
const isNumeric = (value) => {
  return !isNaN(value)
}

// Helper function to check if a credit card number has a valid length
const hasValidLength = (creditCardNumber) => {
  const creditCardStringLength = creditCardNumber.length
  return creditCardStringLength >= 13 && creditCardStringLength <= 16
}

// Helper function to check if a credit card number starts with a valid substring
const startsWithValidSubString = (creditCardNumber) => {
  const validStartSubStrings = ['4', '5', '6', '37', '34', '35']
  return validStartSubStrings.some(subString => creditCardNumber.startsWith(subString))
}

// Helper function to calculate the Luhn validation sum
const calculateLuhnValidationSum = (creditCardNumber) => {
  let validationSum = 0
  creditCardNumber.split('').forEach((digit, index) => {
    let currentDigit = parseInt(digit)
    if (index % 2 === 0) {
      // Multiply every 2nd digit from the left by 2
      currentDigit *= 2
      // if product is greater than 10 add the individual digits of the product to get a single digit
      if (currentDigit > 9) {
        currentDigit %= 10
        currentDigit += 1
      }
    }
    validationSum += currentDigit
  })
  return validationSum
}

const validateCreditCard = (creditCardNumber) => {
  if (typeof creditCardNumber !== 'string') {
    throw new TypeError('The given value is not a string')
  }

  if (!isNumeric(creditCardNumber)) {
    throw new TypeError(`${creditCardNumber} is an invalid credit card number because it has nonnumerical characters.`)
  }

  if (!hasValidLength(creditCardNumber)) {
    throw new Error(`${creditCardNumber} is an invalid credit card number because of its length.`)
  }

  if (!startsWithValidSubString(creditCardNumber)) {
    throw new Error(`${creditCardNumber} is an invalid credit card number because of its first two digits.`)
  }
  
  const validationSum = calculateLuhnValidationSum(creditCardNumber)
  if (validationSum % 10 !== 0) {
    throw new Error(`${creditCardNumber} is an invalid credit card number because it fails the Luhn check.`)
  }

  return true
}

export { validateCreditCard }
// ```

// By extracting helper functions, using early returns, and providing descriptive variable names, the code becomes more modular, readable, and maintainable.

