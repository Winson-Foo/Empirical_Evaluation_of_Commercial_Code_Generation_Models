package java_programs;

import java.util.Arrays;

public class FindFirstInSorted {

    /**
     * Searches for the first occurrence of a given element in a sorted array.
     * 
     * @param sortedArray The sorted array to search in.
     * @param target The element to search for.
     * @return The index of the first occurrence of the element, or -1 if not found.
     */
    public static int findFirstInSorted(int[] sortedArray, int target) {
        int low = 0;
        int high = sortedArray.length - 1;

        while (low <= high) {
            int mid = low + (high - low) / 2;

            if (target == sortedArray[mid] && (mid == 0 || target != sortedArray[mid-1])) {
                return mid;
            } else if (target <= sortedArray[mid]) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }

        return -1;
    }

    /**
     * A simple example usage of the findFirstInSorted method.
     */
    public static void main(String[] args) {
        int[] sortedArray = {1, 2, 3, 3, 4, 4, 5};
        int target = 3;

        int index = findFirstInSorted(sortedArray, target);

        if (index != -1) {
            System.out.println("Target found at index: " + index);
        } else {
            System.out.println("Target not found in the array.");
        }
    }

}