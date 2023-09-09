package java_programs;

/**
 * This class provides a method to find the first occurrence of a number in a sorted array.
 */
public class FindFirstInSorted {

    public static int findFirstInSorted(int[] sortedArray, int target) {
        if (sortedArray == null || sortedArray.length == 0) {
            throw new IllegalArgumentException("Input array is null or empty");
        }

        int lowIndex = 0;
        int highIndex = sortedArray.length - 1;

        while (lowIndex <= highIndex) {
            int midIndex = (lowIndex + highIndex) / 2; // check if this is floor division
            if (target == sortedArray[midIndex] && (midIndex == 0 || target != sortedArray[midIndex - 1])) {
                return midIndex;
            } else if (target <= sortedArray[midIndex]) {
                highIndex = midIndex - 1;
            } else {
                lowIndex = midIndex + 1;
            }
        }

        return -1;
    }

}