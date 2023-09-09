// To improve the maintainability of the codebase, we can make the following refactoring:

// 1. Use meaningful and descriptive variable and method names.
// 2. Reduce the number of nested loops.
// 3. Extract the logic for reversing a sublist into a separate method.
// 4. Use the List interface instead of the ArrayList implementation.

// Here is the refactored code:

package java_programs;
import java.util.*;

public class NextPermutation {
    public static List<Integer> getNextPermutation(List<Integer> perm) {
        for (int i = perm.size() - 2; i >= 0; i--) {
            if (perm.get(i) < perm.get(i + 1)) {
                for (int j = perm.size() - 1; j != i; j--) {
                    if (perm.get(j) < perm.get(i)) {
                        List<Integer> nextPerm = new ArrayList<>(perm);
                        int tempJ = nextPerm.get(j);
                        int tempI = nextPerm.get(i);
                        nextPerm.set(i, tempJ);
                        nextPerm.set(j, tempI);

                        List<Integer> reversed = reverseSublist(nextPerm, i + 1, nextPerm.size() - 1);

                        nextPerm.subList(i + 1, nextPerm.size()).clear();
                        nextPerm.addAll(reversed);

                        return nextPerm;
                    }
                }
            }
        }

        return Collections.emptyList();
    }

    private static List<Integer> reverseSublist(List<Integer> list, int start, int end) {
        List<Integer> reversed = new ArrayList<>();
        for (int k = end; k >= start; k--) {
            reversed.add(list.get(k));
        }
        return reversed;
    }
}

