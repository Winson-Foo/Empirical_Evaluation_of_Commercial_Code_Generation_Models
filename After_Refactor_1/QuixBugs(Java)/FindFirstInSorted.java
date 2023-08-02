package java_programs;

public class FindFirstInSorted {

    /**
     * Find the first occurrence of a number in a sorted array.
     *
     * @param sortedArray The sorted array to search in.
     * @param target      The number to find.
     * @return The index of the first occurrence of the target number, or -1 if not found.
     */
    public static int findFirstInSorted(int[] sortedArray, int target) {
        int low = 0;
        int high = sortedArray.length - 1;

        while (low <= high) {
            int mid = low + (high - low) / 2;

            if (sortedArray[mid] == target && (mid == 0 || sortedArray[mid - 1] != target)) {
                return mid;
            } else if (sortedArray[mid] < target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }

        return -1;
    }

}