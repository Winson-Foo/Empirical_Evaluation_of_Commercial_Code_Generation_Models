// To improve the maintainability of the codebase, here are a few suggestions:

// 1. Add comments to clarify the intention and behavior of the code.
// 2. Use meaningful variable and function names that accurately describe their purpose.
// 3. Break down the code into smaller, more manageable functions that perform specific tasks.
// 4. Use a more descriptive function signature.

// Here's the refactored code implementing these suggestions:

// ```javascript
// Converts a decimal number to binary
function decimalToBinary(decimalNumber) {
  const binaryArray = [];
  
  // Continue looping until the decimal number becomes 0
  while (decimalNumber > 0) {
    // Get the remainder when dividing the decimal number by 2
    const remainder = decimalNumber % 2;
    
    // Add the remainder to the beginning of the binary array
    binaryArray.unshift(remainder);
    
    // Divide the decimal number by 2
    decimalNumber >>= 1;
  }
  
  // Convert the binary array to a string and return it
  return binaryArray.join('');
}

export { decimalToBinary };

// Example usage:
// console.log(decimalToBinary(2)); // Output: '10'
// console.log(decimalToBinary(7)); // Output: '111'
// console.log(decimalToBinary(35)); // Output: '100011'
// ```

// By implementing these changes, the code becomes more readable, easier to understand, and easier to modify in the future if needed.

