// To improve the maintainability of the codebase, we can make a few changes:

// 1. Improve variable naming: Instead of using generic variable names like `left` and `right`, we can use more descriptive names like `startIndex` and `endIndex`.

// 2. Use a for loop instead of a while loop: Instead of using a `while` loop with manual increments and decrements, we can use a `for` loop to iterate over the characters in the string.

// 3. Add a check for empty string: We can add a check at the beginning of the function to return `true` if the input string is empty.

// Here's the refactored code with the above improvements:

// ```javascript
/**
 * @param {string} string
 * @return {boolean}
 */
export default function isPalindrome(string) {
  // Check for empty string
  if (!string.length) {
    return true;
  }

  // Use descriptive variable names
  let startIndex = 0;
  let endIndex = string.length - 1;

  for (; startIndex < endIndex; startIndex++, endIndex--) {
    if (string[startIndex] !== string[endIndex]) {
      return false;
    }
  }

  return true;
}
// ```

// By making these changes, the code is more readable and maintainable. The variable names clearly indicate their purpose, and the use of a `for` loop makes the logic more concise.

