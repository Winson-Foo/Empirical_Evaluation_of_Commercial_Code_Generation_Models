package java_programs;
import java.util.*;

/**
 * This class calculates the maximum sublist sum of an array.
 */
public class MaxSublistSum {

    /**
     * Calculates the maximum sublist sum of the given array.
     * @param arr the input array
     * @return the maximum sublist sum
     */
    public static int calculateMaxSublistSum(int[] arr) {
        int maxEndingHere = 0;
        int maxSoFar = 0;

        for (int x : arr) {
            maxEndingHere = maxEndingHere + x;
            maxSoFar = Math.max(maxSoFar, maxEndingHere);
        }

        return maxSoFar;
    }
}