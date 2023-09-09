// To improve the maintainability of the codebase, we can divide the functionality into smaller, more manageable methods. Here is the refactored code with improved maintainability:

package java_programs;
import java.util.ArrayList;

public class NextPermutation {

    public static ArrayList<Integer> nextPermutation(ArrayList<Integer> perm) {
        int i = findDecreasingIndex(perm);
        if (i == -1) {
            return new ArrayList<Integer>();
        }
        
        int j = findNextLargerIndex(perm, i);
        swapElements(perm, i, j);
        reverseRemaining(perm, i + 1);
        
        return perm;
    }
    
    private static int findDecreasingIndex(ArrayList<Integer> perm) {
        for (int i = perm.size() - 2; i >= 0; i--) {
            if (perm.get(i) < perm.get(i + 1)) {
                return i;
            }
        }
        return -1;
    }
    
    private static int findNextLargerIndex(ArrayList<Integer> perm, int i) {
        int target = perm.get(i);
        for (int j = perm.size() - 1; j > i; j--) {
            if (perm.get(j) > target) {
                return j;
            }
        }
        return -1;
    }
    
    private static void swapElements(ArrayList<Integer> perm, int i, int j) {
        int temp = perm.get(i);
        perm.set(i, perm.get(j));
        perm.set(j, temp);
    }
    
    private static void reverseRemaining(ArrayList<Integer> perm, int startIndex) {
        int endIndex = perm.size() - 1;
        while (startIndex < endIndex) {
            swapElements(perm, startIndex, endIndex);
            startIndex++;
            endIndex--;
        }
    }
}

// By breaking the functionality into smaller methods, it becomes easier to understand and modify specific parts of the code. This also improves readability and maintainability by promoting code reuse and separation of concerns.

