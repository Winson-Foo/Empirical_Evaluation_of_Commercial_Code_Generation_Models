// To improve the maintainability of the codebase, we can make the following changes:

// 1. Use destructuring assignment to swap array elements instead of using a temporary variable.
// 2. Use more descriptive variable and function names.
// 3. Add proper comments to explain the purpose of each section of code.

// Here's the refactored code:

// ```javascript
export const shuffleArray = (array) => {
  let maxLength = array.length;

  // While there remain elements to shuffle...
  while (maxLength) {
    // Pick a remaining element randomly...
    const randomIdx = Math.floor(Math.random() * maxLength--);

    // Swap the elements using destructuring assignment
    [array[maxLength], array[randomIdx]] = [array[randomIdx], array[maxLength]];
  }

  return array;
}
// ```

// By following these changes, the code becomes more readable, maintainable, and self-explanatory.

