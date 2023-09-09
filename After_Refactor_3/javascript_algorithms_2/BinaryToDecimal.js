// To improve the maintainability of the code, you can follow these steps:

// 1. Break down the logic into smaller, self-contained functions.
// 2. Use meaningful variable and function names.
// 3. Add comments and documentation to explain the purpose of the code.
// 4. Use ES6 features like arrow functions and const/let instead of var.

// Here's the refactored code:

// ```javascript
export default function binaryToDecimal(binaryString) {
  const binaryDigits = splitBinaryIntoDigits(binaryString);
  const decimalNumber = calculateDecimalNumber(binaryDigits);
  return decimalNumber;
}

function splitBinaryIntoDigits(binaryString) {
  return binaryString.split('').reverse();
}

function calculateDecimalNumber(binaryDigits) {
  let decimalNumber = 0;
  binaryDigits.forEach((binaryDigit, index) => {
    decimalNumber += binaryDigit * (Math.pow(2, index));
  });
  return decimalNumber;
}
// ```

// By refactoring the code in this way, it becomes easier to understand and maintain. Each function has a clear purpose, making it easier to make changes or add new functionality in the future.

