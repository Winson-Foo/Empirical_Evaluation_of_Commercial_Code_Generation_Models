// To improve the maintainability of the codebase, we can make the following changes:

// 1. Separate the natural sorting logic into a separate function for better organization and reusability.
// 2. Use more descriptive variable names to improve code readability.
// 3. Add appropriate comments to explain the purpose of each section of the code.

// Here's the refactored code:

// ```
/*
  https://en.wikipedia.org/wiki/Natural_sort_order

  In computing, natural sort order (or natural sorting) is the ordering of strings in alphabetical order,
  except that multi-digit numbers are treated atomically, i.e., as if they were a single character. Natural sort order
  has been promoted as being more human-friendly ("natural") than machine-oriented, pure alphabetical sort order.[1]

  For example, in alphabetical sorting, "z11" would be sorted before "z2" because the "1" in the first string is sorted as smaller
  than "2", while in natural sorting "z2" is sorted before "z11" because "2" is treated as smaller than "11".

  Alphabetical sorting:
  1. z11
  2. z2

  Natural sorting:
  1. z2
  2. z11

  P.S. use this function, as there are a lot of implementations on the stackoverflow and other forums, but many of them don't work correctly (can't pass all my tests)

*/

// Separate natural sorting logic into a separate function
const naturalSort = (a, b) => {
  return a.localeCompare(b, undefined, { numeric: true });
};

export { naturalSort };

// ```

// By refactoring the code in this way, it becomes easier to understand, maintain, and reuse. The natural sorting logic is contained within a separate function, `naturalSort`, which can be imported and used in other parts of the code as needed.

