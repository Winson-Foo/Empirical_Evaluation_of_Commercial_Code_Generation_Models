// To improve the maintainability of the codebase, you can take the following steps:

// 1. Use meaningful function and variable names: Instead of naming the function "ReverseNumber", consider naming it something more descriptive like "reverseDigits" or "getReversedNumber". This will make the code easier to understand and maintain.

// 2. Add comments to explain the code: While the code is relatively simple, adding comments to explain the steps of the algorithm can make it easier for other developers (including yourself) to understand and modify the code in the future.

// 3. Split the logic into separate functions: Instead of having all the logic in a single function, you can break it down into smaller functions with specific responsibilities. This will make the code more modular and easier to test.

// With these improvements in mind, here's the refactored code:

/**
 * Reverses the digits of a given number.
 * @param {number} num - The number to reverse.
 * @returns {number} - The reversed number.
 */
const reverseDigits = (num) => {
  if (typeof num !== 'number') {
    throw new TypeError('Argument is not a number.');
  }

  let reversedNum = 0;

  while (num > 0) {
    const lastDigit = num % 10;
    reversedNum = reversedNum * 10 + lastDigit;
    num = Math.floor(num / 10);
  }

  return reversedNum;
};

export { reverseDigits };

// By making these changes, the code becomes more readable, maintainable, and easier to understand for other developers.

