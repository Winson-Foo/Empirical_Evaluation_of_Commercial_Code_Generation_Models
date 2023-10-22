// To improve the maintainability of this codebase, the following modifications can be made:

// 1. Add comments to indicate the purpose of each section of code.
// 2. Use meaningful variable names to enhance code understanding.
// 3. Extract magic numbers and hard-coded values into named constants.
// 4. Format the code to follow proper indentation and coding standards.

// Here is the refactored code:

// ```java
package correct_java_programs;

/**
 * This class contains a method to find the first occurrence of a number in a sorted array.
 */
public class FIND_FIRST_IN_SORTED {

    /**
     * Finds the first occurrence of a number in a sorted array.
     *
     * @param arr The sorted array to search in.
     * @param x The number to search for.
     * @return The index of the first occurrence of the number in the array, -1 if not found.
     */
    public static int findFirstInSorted(int[] arr, int x) {
        int low = 0;
        int high = arr.length;

        while (low < high) {
            int mid = (low + high) / 2;

            if (x == arr[mid] && (mid == 0 || x != arr[mid - 1])) {
                return mid;
            } else if (x <= arr[mid]) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        return -1;
    }
}
// ```

// By making these changes, the code becomes more readable, maintainable, and easier to understand for future developers.

