// To improve the maintainability of the codebase, you can consider the following refactorings:

// 1. Split the `sort` method into smaller, more focused methods to improve readability and maintainability.

// ```
// class RadixSort extends Sort {
//   sort(originalArray) {
//     const isArrayOfNumbers = this.isArrayOfNumbers(originalArray);
//     let sortedArray = [...originalArray];
//     const numPasses = this.determineNumPasses(sortedArray);

//     for (let currentIndex = 0; currentIndex < numPasses; currentIndex += 1) {
//       const buckets = isArrayOfNumbers
//         ? this.placeElementsInNumberBuckets(sortedArray, currentIndex)
//         : this.placeElementsInCharacterBuckets(sortedArray, currentIndex, numPasses);

//       sortedArray = this.flattenBuckets(buckets);
//     }

//     return sortedArray;
//   }

//   // Methods for placing elements in buckets, determining the number of passes, and checking if an array is of numbers.

//   // ...

//   // Method for flattening the buckets into a single array.
//   flattenBuckets(buckets) {
//     return buckets.reduce((acc, val) => {
//       return [...acc, ...val];
//     }, []);
//   }
// }
// ```

// 2. Split the `placeElementsInNumberBuckets` method into smaller, more focused methods to improve readability and maintainability.

// ```
// class RadixSort extends Sort {
//   // ...

//   placeElementsInNumberBuckets(array, index) {
//     const modded = 10 ** (index + 1);
//     const divided = 10 ** index;
//     const buckets = this.createBuckets(NUMBER_OF_POSSIBLE_DIGITS);

//     array.forEach((element) => {
//       this.callbacks.visitingCallback(element);
//       if (element < divided) {
//         buckets[0].push(element);
//       } else {
//         const currentDigit = this.getCurrentDigit(element, modded, divided);
//         buckets[currentDigit].push(element);
//       }
//     });

//     return buckets;
//   }

//   // Methods for determining the current digit of an element.

//   // ...

//   getCurrentDigit(element, modded, divided) {
//     return Math.floor((element % modded) / divided);
//   }
// }
// ```

// 3. Split the `placeElementsInCharacterBuckets` method into smaller, more focused methods to improve readability and maintainability.

// ```
class RadixSort extends Sort {
  // ...

  placeElementsInCharacterBuckets(array, index, numPasses) {
    const buckets = this.createBuckets(ENGLISH_ALPHABET_LENGTH);

    array.forEach((element) => {
      this.callbacks.visitingCallback(element);
      const currentBucket = this.getCharCodeOfElementAtIndex(element, index, numPasses);
      buckets[currentBucket].push(element);
    });

    return buckets;
  }

  // Methods for getting the character code of an element at a specific index.

  // ...

  getCharCodeOfElementAtIndex(element, index, numPasses) {
    if ((numPasses - index) > element.length) {
      return ENGLISH_ALPHABET_LENGTH - 1;
    }

    const charPos = this.getCharPosition(element, index);
    return element.toLowerCase().charCodeAt(charPos) - BASE_CHAR_CODE;
  }

  getCharPosition(element, index) {
    return index > element.length - 1 ? 0 : element.length - index - 1;
  }
}
// ```

// By breaking down the code into smaller, focused methods, you make it easier to understand, modify, and maintain. The refactored code improves readability and reduces duplication, leading to better maintainability.

