package java_programs;

public class Knapsack {
    public static int solveKnapsack(int knapsackCapacity, int[][] items) {
        int totalWeight = 0, totalValue = 0;
        int numberOfItems = items.length;
        int knapsackMemo[][] = new int[numberOfItems + 1][knapsackCapacity + 1];

        for (int i = 0; i <= numberOfItems; i++) {
            if (i - 1 >= 0) {
                totalWeight = items[i - 1][0];
                totalValue = items[i - 1][1];
            }
            
            for (int j = 0; j <= knapsackCapacity; j++) {
                if (i == 0 || j == 0) {
                    knapsackMemo[i][j] = 0;
                } else if (totalWeight < j) {
                    knapsackMemo[i][j] = Math.max(knapsackMemo[i - 1][j], totalValue + knapsackMemo[i - 1][j - totalWeight]);
                } else {
                    knapsackMemo[i][j] = knapsackMemo[i - 1][j];
                }
            }
        }
        
        return knapsackMemo[numberOfItems][knapsackCapacity];
    }
}