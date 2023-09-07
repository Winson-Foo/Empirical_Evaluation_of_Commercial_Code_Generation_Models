// To improve the maintainability of this codebase, we can apply the following refactorings:

// 1. Move the code to a proper package: It is recommended to move the code to a relevant package that reflects its functionality. For example, the package name can be changed to "shortest_path".

// 2. Use meaningful names: Rename variables and methods to more descriptive names that accurately represent their purpose and functionality. This will make the code easier to understand and maintain.

// 3. Extract magic numbers to constants: Replace the magic number `INF` with a meaningful constant name. This will make it easier to understand the code and modify the value if needed.

// 4. Use diamond operator: Instead of explicitly mentioning the type arguments in `new ArrayList<>()`, use the diamond operator `new ArrayList<>()`. This will make the code cleaner and easier to read.

// Here's the refactored code with the suggested improvements:

package shortest_path;

import java.util.*;

public class ShortestPathLengths {
    private static final int INFINITY = 99999;

    public static Map<List<Integer>, Integer> shortestPathLengths(int numNodes, Map<List<Integer>, Integer> lengthByEdge) {
        Map<List<Integer>, Integer> lengthByPath = new HashMap<>();
        
        for (int i = 0; i < numNodes; i++) {
            for (int j = 0; j < numNodes; j++) {
                List<Integer> edge = new ArrayList<>();
                edge.add(i);
                edge.add(j);

                if (i == j) {
                    lengthByPath.put(edge, 0);
                } else if (lengthByEdge.containsKey(edge)) {
                    lengthByPath.put(edge, lengthByEdge.get(edge));
                } else {
                    lengthByPath.put(edge, INFINITY);
                }
            }
        }

        for (int k = 0; k < numNodes; k++) {
            for (int i = 0; i < numNodes; i++) {
                for (int j = 0; j < numNodes; j++) {
                    int updateLength = Math.min(
                        lengthByPath.get(Arrays.asList(i, j)),
                        sumLengths(lengthByPath.get(Arrays.asList(i, k)), lengthByPath.get(Arrays.asList(j, k)))
                    );
                    lengthByPath.put(Arrays.asList(i, j), updateLength);
                }
            }
        }
        
        return lengthByPath;
    }

    private static int sumLengths(int a, int b) {
        if (a == INFINITY || b == INFINITY) {
            return INFINITY;
        }
        return a + b;
    }
}

// These changes will make the codebase more maintainable, easier to understand, and follow best coding practices.

