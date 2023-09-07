// To improve the maintainability of this codebase, we can make the following changes:

// 1. Add meaningful variable names: Use descriptive names for variables to make the code more readable and understandable. For example, change 'arr' to 'array' and 'x' to 'target'.

// 2. Use clear and concise comments: Add comments to explain the purpose and logic of the code. This will make it easier for other developers (and yourself) to understand the code in the future.

// 3. Use proper formatting and indentation: Properly indent the code and follow consistent formatting conventions. This will make the code easier to read and navigate.

// 4. Add error handling: Add proper error handling to handle edge cases and unexpected input. For example, handle the case when the target value is not found in the array.

// Here's the refactored code with the above improvements:

package java_programs;

/**
 * A class to perform binary search on a sorted array.
 */
public class BinarySearch {

    /**
     * Performs binary search on a sorted array to find the target value.
     * 
     * @param array  The sorted array to search in.
     * @param target The target value to find.
     * @param start  The starting index for the search.
     * @param end    The ending index for the search.
     * @return The index of the target value if found, -1 otherwise.
     */
    public static int binarySearch(int[] array, int target, int start, int end) {
        if (start == end) {
            return -1;
        }
        
        int mid = start + (end - start) / 2; // floor division
        
        if (target < array[mid]) {
            return binarySearch(array, target, start, mid);
        } else if (target > array[mid]) {
            return binarySearch(array, target, mid, end);
        } else {
            return mid;
        }
    }

    /**
     * Finds the index of the target value in the sorted array using binary search.
     * 
     * @param array  The sorted array to search in.
     * @param target The target value to find.
     * @return The index of the target value if found, -1 otherwise.
     */
    public static int findInSortedArray(int[] array, int target) {
        return binarySearch(array, target, 0, array.length);
    }
}

