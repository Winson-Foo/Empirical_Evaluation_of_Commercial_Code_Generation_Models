// To improve the maintainability of the codebase, we can make the following changes:

// 1. Add meaningful comments to explain the purpose of each function and code block.
// 2. Use descriptive variable names that accurately represent their purpose.
// 3. Use consistent indentation and formatting to improve readability.
// 4. Remove unnecessary checks and optimizations that do not affect the logic of the code.
// 5. Extract helper functions to handle specific tasks and improve code modularity.

// Here's the refactored code:

/**
 * Given two strings s1 and s2 of the same length, return true if s2 is a scrambled string of s1, otherwise, return false.
 * @param {string} s1
 * @param {string} s2
 * @return {boolean}
 */

const isScramble = (s1, s2) => {
  // Create a memoization object to store results of subproblems
  const dp = {}
  return isScrambleHelper(dp, s1, s2)
}

const isScrambleHelper = function(dp, s1, s2) {
  // If the result for the current input has already been calculated, return it from memoization
  if (dp[s1 + s2] !== undefined) return dp[s1 + s2]

  // Base case: if the strings are equal, return true
  if (s1 === s2) return true

  // Check if the characters in s1 and s2 are the same
  const characterCount = {}
  for (let i = 0; i < s1.length; i++) {
    const char1 = s1[i]
    const char2 = s2[i]

    // Increment count for char1 in characterCount
    if (characterCount[char1] === undefined) characterCount[char1] = 0
    characterCount[char1]++

    // Decrement count for char2 in characterCount
    if (characterCount[char2] === undefined) characterCount[char2] = 0
    characterCount[char2]--
  }

  // If characterCount has any non-zero values, the strings are not anagrams, so return false
  for (const count of Object.values(characterCount)) {
    if (count !== 0) {
      dp[s1 + s2] = false
      return false
    }
  }

  // Check all possible splits of s1 and s2 and recursively check if they are scrambles
  for (let i = 1; i < s1.length; i++) {
    const isScramble1 = (
      isScrambleHelper(dp, s1.substr(0, i), s2.substr(0, i)) &&
      isScrambleHelper(dp, s1.substr(i), s2.substr(i))
    )
    const isScramble2 = (
      isScrambleHelper(dp, s1.substr(0, i), s2.substr(s2.length - i)) &&
      isScrambleHelper(dp, s1.substr(i), s2.substr(0, s2.length - i))
    )

    if (isScramble1 || isScramble2) {
      dp[s1 + s2] = true
      return true
    }
  }

  // If no valid split found, store false in memoization and return false
  dp[s1 + s2] = false
  return false
}

export { isScramble }

