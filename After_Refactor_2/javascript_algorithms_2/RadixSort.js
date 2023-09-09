// To improve the maintainability of this codebase, we can follow these steps:

// 1. Add comments to describe the purpose and functionality of each section of code.

// 2. Use meaningful variable names that accurately describe their purpose.

// 3. Extract helper functions for reusable code.

// 4. Remove unnecessary or redundant code.

// Here is the refactored code with the above improvements:

/*
* Radix sorts an integer array without comparing the integers.
* It groups the integers by their digits which share the same
* significant position.
* For more information see: https://en.wikipedia.org/wiki/Radix_sort
*/
export function radixSort (items, RADIX) {
  // Default radix is 10 because we usually count in base 10.
  if (RADIX === undefined || RADIX < 1) {
    RADIX = 10;
  }

  let maxLength = false;
  let placement = 1;

  // Iterate until all digits have been processed.
  while (!maxLength) {
    maxLength = true;

    // Create buckets for each possible digit.
    const buckets = createBuckets(RADIX);

    // Group items into buckets based on the current digit.
    for (let j = 0; j < items.length; j++) {
      const current = items[j];
      const currentDigit = Math.floor(current / placement) % RADIX;
      buckets[currentDigit].push(current);
      if (maxLength && current > 0) {
        maxLength = false;
      }
    }

    // Update the original array with the items from the buckets.
    updateArray(items, buckets);

    placement *= RADIX;
  }

  return items;
}

/**
 * Creates an array of empty buckets for radix sorting.
 * @param {number} RADIX - The radix/base to use.
 * @returns {Array[]} - Array of empty buckets.
 */
function createBuckets(RADIX) {
  const buckets = [];
  for (let i = 0; i < RADIX; i++) {
    buckets.push([]);
  }
  return buckets;
}

/**
 * Updates the original array with the items from the buckets.
 * @param {number[]} items - The original array to update.
 * @param {Array[]} buckets - The buckets with items to merge.
 */
function updateArray(items, buckets) {
  let index = 0;
  for (const bucket of buckets) {
    for (const item of bucket) {
      items[index] = item;
      index++;
    }
  }
}

// By following these improvements, the code is more readable, maintainable, and modular.

