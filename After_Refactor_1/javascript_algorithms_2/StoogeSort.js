// To improve the maintainability of this codebase, we can make the following changes:

// 1. Rename the function `stoogeSort` to a more descriptive name, such as `sortArray`.
// 2. Add comments to clarify the purpose and functionality of the code.
// 3. Extract the swap functionality into a separate function.
// 4. Rename the variables to have more meaningful names.
// 5. Use proper indentation and formatting.

// Here is the refactored code:

export function sortArray(items, leftIndex, rightIndex) {
  // Swap two elements in the array
  function swap(arr, i, j) {
    const temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp
  }
  
  // Check if left element is greater than right element, swap if necessary
  if (items[rightIndex - 1] < items[leftIndex]) {
    swap(items, leftIndex, rightIndex - 1)
  }
  
  const length = rightIndex - leftIndex
  
  // Recursively divide array until length is greater than 2
  if (length > 2) {
    const third = Math.floor(length / 3)
    
    // Sort the left two-thirds of the array
    sortArray(items, leftIndex, rightIndex - third)
    
    // Sort the right two-thirds of the array
    sortArray(items, leftIndex + third, rightIndex)
    
    // Sort the left two-thirds of the array again to ensure correct order
    sortArray(items, leftIndex, rightIndex - third)
  }
  
  return items
}

