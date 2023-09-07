package java_programs;

public class MaxSublistSum {
    public static int getMaxSublistSum(int[] arr) {
        int maxEndingHere = 0;
        int maxSoFar = 0;

        for (int num : arr) {
            maxEndingHere = Math.max(0, maxEndingHere + num);
            maxSoFar = Math.max(maxSoFar, maxEndingHere);
        }

        return maxSoFar;
    }
}