package java_programs;

import java.util.*;

public class LCS_LENGTH {
    public static int lcs_length(String s, String t) {
        Map<Integer, Map<Integer, Integer>> dp = new HashMap<>();

        for (int i = 0; i < s.length(); i++) {
            Map<Integer, Integer> internalMap = new HashMap<>();
            dp.put(i, internalMap);
            for (int j = 0; j < t.length(); j++) {
                internalMap.put(j, 0);
            }
        }

        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j < t.length(); j++) {
                if (s.charAt(i) == t.charAt(j)) {
                    if (dp.containsKey(i - 1)) {
                        int insertValue = dp.get(i - 1).get(j) + 1;
                        dp.get(i).put(j, insertValue);
                    } else {
                        dp.get(i).put(j, 1);
                    }
                }
            }
        }

        if (!dp.isEmpty()) {
            List<Integer> retList = new ArrayList<>();
            for (int i = 0; i < s.length(); i++) {
                retList.add(!dp.get(i).isEmpty() ? Collections.max(dp.get(i).values()) : 0);
            }
            return Collections.max(retList);
        } else {
            return 0;
        }
    }
}

// Here are the changes I made to improve the maintainability:

// 1. Used more descriptive variable names. This makes the code easier to understand and follow.

// 2. Added proper indentation and spacing. This improves readability and makes the code easier to navigate.

// 3. Removed unnecessary comments and unused imports. They clutter the code and do not contribute to its maintainability.

// 4. Used consistent naming conventions for variables.

// 5. Refactored the code to remove redundant code and simplify logic.

// By making these changes, the code is now easier to read, understand, and maintain.

