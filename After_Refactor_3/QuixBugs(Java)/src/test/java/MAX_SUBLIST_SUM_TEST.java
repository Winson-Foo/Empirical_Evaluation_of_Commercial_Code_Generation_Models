package test.java;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import correct_java_programs.MAX_SUBLIST_SUM;

public class MAX_SUBLIST_SUM_TEST {
    @Test
    public void test_0() throws java.lang.Exception {
        int result = MAX_SUBLIST_SUM.calculateMaxSublistSum(new int[]{4,-5,2,1,-1,3});
        assertEquals( (int) 5, result);
    }

    @Test
    public void test_1() throws java.lang.Exception {
        int result = MAX_SUBLIST_SUM.calculateMaxSublistSum(new int[]{0,-1,2,-1,3,-1,0});
        assertEquals( (int) 4, result);
    }

    @Test
    public void test_2() throws java.lang.Exception {
        int result = MAX_SUBLIST_SUM.calculateMaxSublistSum(new int[]{3,4,5});
        assertEquals( (int) 12, result);
    }

    @Test
    public void test_3() throws java.lang.Exception {
        int result = MAX_SUBLIST_SUM.calculateMaxSublistSum(new int[]{4,-2,-8,5,-2,7,7,2,-6,5});
        assertEquals( (int) 19, result);
    }

    @Test
    public void test_4() throws java.lang.Exception {
        int result = MAX_SUBLIST_SUM.calculateMaxSublistSum(new int[]{-4,-4,-5});
        assertEquals( (int) 0, result);
    }

    @Test
    public void test_5() throws java.lang.Exception {
        int result = MAX_SUBLIST_SUM.calculateMaxSublistSum(new int[]{-2,1,-3,4,-1,2,1,-5,4});
        assertEquals( (int) 6, result);
    }
}