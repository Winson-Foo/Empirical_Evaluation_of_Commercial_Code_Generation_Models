// To improve the maintainability of the codebase, we can refactor the code as follows:

/**
 * @param {string} string
 * @return {boolean}
 */
export default function isPalindrome(string) {
  const length = string.length;
  for (let i = 0; i < length / 2; i++) {
    if (string[i] !== string[length - 1 - i]) {
      return false;
    }
  }
  return true;
}

// In the refactored code:
// - The variables 'left' and 'right' have been replaced with a single loop variable 'i'.
// - The loop condition has been updated to iterate until half of the length of the string, rather than checking 'left < right' in each iteration.
// - The variables 'left' and 'right' have been replaced with 'i' and 'length - 1 - i' respectively, to access the characters from both ends of the string.
// - The 'left += 1' and 'right -= 1' statements have been removed.
// - The refactored code is more concise and easier to read, reducing maintenance efforts.

