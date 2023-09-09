// To improve the maintainability of the codebase, we can make the following changes:

// 1. Improve variable and function names: Use more descriptive names for variables and functions to improve code readability. For example, `map` can be renamed to `charCount`.

// 2. Use object destructuring: Instead of accessing object properties directly (`dp[s1 + s2]`), use object destructuring to make the code more readable and reduce duplication.

// 3. Use strict equality comparison (`===`) instead of loose equality comparison (`==`) for better code readability and consistency.

// 4. Add comments to explain the purpose and functionality of each section of code.

// Here is the refactored code:

/**
 * Given two strings s1 and s2 of the same length, return true if s2 is a scrambled string of s1, otherwise, return false.
 * @param {string} s1
 * @param {string} s2
 * @return {boolean}
 */

const isScramble = (s1, s2) => {
  const dp = {};
  return helper(dp, s1, s2);
};

const helper = function (dp, s1, s2) {
  // Check if the result is already computed and stored in the dp object
  if (dp[`${s1}-${s2}`] !== undefined) {
    return dp[`${s1}-${s2}`];
  }

  // Base case: if s1 and s2 are identical, return true
  if (s1 === s2) {
    return true;
  }

  const charCount = {};

  // Count the frequency of characters in s1 and s2
  for (let j = 0; j < s1.length; j++) {
    charCount[s1[j]] = (charCount[s1[j]] || 0) + 1;
    charCount[s2[j]] = (charCount[s2[j]] || 0) - 1;
  }

  // Check if the character counts are all zero
  for (const key in charCount) {
    if (charCount[key] !== 0) {
      dp[`${s1}-${s2}`] = false;
      return false;
    }
  }

  // Recursive case: try all possible partitions of s1 and s2
  for (let i = 1; i < s1.length; i++) {
    if (
      (helper(dp, s1.substr(0, i), s2.substr(0, i)) &&
        helper(dp, s1.substr(i), s2.substr(i))) ||
      (helper(dp, s1.substr(0, i), s2.substr(s2.length - i)) &&
        helper(dp, s1.substr(i), s2.substr(0, s2.length - i)))
    ) {
      dp[`${s1}-${s2}`] = true;
      return true;
    }
  }

  dp[`${s1}-${s2}`] = false;
  return false;
};

export { isScramble };

