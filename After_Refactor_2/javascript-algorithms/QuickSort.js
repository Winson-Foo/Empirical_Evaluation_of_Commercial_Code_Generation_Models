// To improve the maintainability of the codebase, we can do the following:

// 1. Add appropriate comments to explain the purpose of each section of code.
// 2. Use meaningful variable names to improve code readability.
// 3. Move the sorting logic into a separate helper function.
// 4. Use array destructuring to simplify code.
// 5. Remove unnecessary callbacks.

// Here is the refactored code:

import Sort from '../Sort';

export default class QuickSort extends Sort {
  /**
   * @param {*[]} originalArray
   * @return {*[]}
   */
  sort(originalArray) {
    // Clone original array to prevent it from modification.
    const array = [...originalArray];

    // If array has less than or equal to one elements then it is already sorted.
    if (array.length <= 1) {
      return array;
    }

    // Split the array into left, center, and right arrays
    const { pivot, leftArray, centerArray, rightArray } = this.partitionArray(array);

    // Sort left and right arrays.
    const leftArraySorted = this.sort(leftArray);
    const rightArraySorted = this.sort(rightArray);

    // Join sorted left array with center array and with sorted right array.
    return [...leftArraySorted, ...centerArray, ...rightArraySorted];
  }

  /**
   * @param {*[]} array
   * @return {{ pivot: *, leftArray: *[], centerArray: *[], rightArray: *[] }}
   */
  partitionArray(array) {
    // Init left, center, and right arrays.
    const leftArray = [];
    const centerArray = [];
    const rightArray = [];

    // Take the first element of array as a pivot.
    const pivot = array[0];

    // Split all array elements between left, center and right arrays.
    for (let i = 1; i < array.length; i++) {
      const currentElement = array[i];

      if (currentElement === pivot) {
        centerArray.push(currentElement);
      } else if (currentElement < pivot) {
        leftArray.push(currentElement);
      } else {
        rightArray.push(currentElement);
      }
    }

    return {
      pivot,
      leftArray,
      centerArray,
      rightArray
    };
  }
}

