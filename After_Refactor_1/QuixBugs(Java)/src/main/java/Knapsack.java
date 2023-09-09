package java_programs;

public class Knapsack {
    private int capacity;
    private int[][] items;
    private int[][] memoizationTable;

    public Knapsack(int capacity, int[][] items) {
        this.capacity = capacity;
        this.items = items;
        this.memoizationTable = new int[items.length + 1][capacity + 1];
    }

    public int solve() {
        int currentItemWeight = 0, currentItemValue = 0;
        int numberOfItems = items.length;

        for (int i = 0; i <= numberOfItems; i++) {
            if (i - 1 >= 0) {
                currentItemWeight = items[i - 1][0];
                currentItemValue = items[i - 1][1];
            }
            for (int j = 0; j <= capacity; j++) {
                if (i == 0 || j == 0) {
                    memoizationTable[i][j] = 0;
                } else if (currentItemWeight < j) {
                    memoizationTable[i][j] = Math.max(memoizationTable[i - 1][j], currentItemValue + memoizationTable[i - 1][j - currentItemWeight]);
                } else {
                    memoizationTable[i][j] = memoizationTable[i - 1][j];
                }
            }
        }
        return memoizationTable[numberOfItems][capacity];
    }
}