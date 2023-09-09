package java_programs;
import java.util.*;

public class MaxSublistSum {

    public static int findMaxSublistSum(int[] arr) {
        int maxEndingHere = 0;
        int maxSoFar = 0;

        for (int x : arr) {
            maxEndingHere = Math.max(x, maxEndingHere + x);
            maxSoFar = Math.max(maxSoFar, maxEndingHere);
        }

        return maxSoFar;
    }
}

